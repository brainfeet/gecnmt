(ns gecnmt.prepare
  (:require [clojure.java.io :as io]
            [clojure.java.shell :as sh]
            [clojure.set :as set]
            [clojure.string :as str]
            [aid.core :as aid]
            [cats.monad.either :as either]
            [cheshire.core :refer :all]
            [com.rpl.specter :as s]
            [me.raynes.fs :as fs]
            [gecnmt.command :as command]))

(def join-paths
  (comp str
        io/file))

(def get-dataset-path
  (partial join-paths "resources/dataset"))

(def extract
  (partial command/python
           "bin/WikiExtractor.py"
           "--json"
           "-o"
           (get-dataset-path "simple/extracted")
           (get-dataset-path "simple/original.xml")))

(def parse-extracted
  (comp (partial mapcat (comp (partial str/join "\n")
                              (partial remove str/blank?)
                              str/split-lines
                              :text
                              (partial (aid/flip parse-string) true)))
        str/split-lines))

(defn slurp-extracted
  []
  (->> #"^wiki_\d{2}"
       (fs/find-files (get-dataset-path "simple/extracted"))
       (map slurp)))

(defn spit-parents
  [f & more]
  (-> f
      fs/parent
      fs/mkdirs)
  (apply spit f more))

(defn appending-spit-parents
  [f content]
  (spit-parents f content :append true))

(def combine
  (comp (partial run! (partial appending-spit-parents
                               (get-dataset-path "simple/combined.txt")))
        (partial mapcat parse-extracted)
        slurp-extracted))

(defn python
  [& more]
  (sh/with-sh-dir "python"
                  (apply command/export
                         "PYTHONPATH=$(pwd)"
                         "&&"
                         "source"
                         "activate"
                         "gecnmt"
                         "&&"
                         "python"
                         more)))

(def parse
  (comp (partial python
                 "gecnmt/parse.py"
                 "--path")
        fs/absolute
        (partial (aid/flip get-dataset-path) "combined.txt")))

(def parse-keywordize
  (partial (aid/flip parse-string) true))

(def is-ascii?
  (partial every? (comp (partial > 128)
                        int)))

(def has-newline?
  (partial re-find #".*\n.*"))

(def split-sentences*
  (comp (partial map
                 (comp prn-str
                       vec
                       (partial filter
                                (comp (aid/build and
                                                 is-ascii?
                                                 (complement has-newline?)
                                                 (complement str/blank?))
                                      :text))
                       flatten))
        (partial partition 2)
        (partial partition-by :is_sent_start)
        (partial s/setval* [s/FIRST :is_sent_start] true)
        parse-keywordize))

(defn make-process-lines
  [{:keys [f input output]}]
  (fn [dataset]
    (with-open [file (->> input
                          (get-dataset-path dataset)
                          io/reader)]
      (->> file
           line-seq
           f
           (run! (partial appending-spit-parents (get-dataset-path dataset
                                                                   output)))))))

(def split-sentences
  (make-process-lines {:f      (partial mapcat split-sentences*)
                       :input  "parsed.txt"
                       :output "split.txt"}))

(def randomize
  (aid/build (partial command/shuf "-o")
             (partial (aid/flip get-dataset-path) "random.txt")
             (partial (aid/flip get-dataset-path) "split.txt")))

(def append-newline
  (partial (aid/flip str) "\n"))

(def get-text
  (make-process-lines {:f      (comp (partial map (comp append-newline
                                                        (partial str/join " ")))
                                     (partial s/transform* [s/ALL s/ALL] :text)
                                     (partial map read-string))
                       :input  "random.txt"
                       :output "text.txt"}))

(defn learn-bpe
  [dataset]
  (command/python "bin/learn_bpe.py"
                  "-s"
                  "10000"
                  "<"
                  (get-dataset-path dataset "text.txt")
                  ">"
                  (get-dataset-path dataset "codes.txt")))

(defn apply-bpe
  [dataset]
  (command/python "bin/apply_bpe.py"
                  "-c"
                  (get-dataset-path dataset "codes.txt")
                  "<"
                  (get-dataset-path dataset "text.txt")
                  ">"
                  (get-dataset-path dataset "bpe.txt")))

(defn build-vocabulary
  [dataset]
  (with-open [f (->> "bpe.txt"
                     (get-dataset-path dataset)
                     io/reader)]
    (->> f
         line-seq
         (mapcat (partial (aid/flip str/split) #" "))
         (reduce conj #{})
         (map-indexed (fn [index word]
                        ;consider EOS and SOS tokens
                        {(+ 2 index) word}))
         (apply merge {0 "<SOS>"
                       1 "<EOS>"})
         ((juxt (comp (partial spit (get-dataset-path dataset "word.json"))
                      generate-string)
                (comp (partial spit (get-dataset-path dataset "index.edn"))
                      set/map-invert))))))

(defn structure
  [{:keys [bpe index random]}]
  (->> bpe
       (map (comp (partial map index)
                  (partial (aid/flip str/split) #" ")))
       (map (fn [tokens bpes]
              {:length      (count bpes)
               :input-bpes  (->> bpes
                                 (s/setval s/BEGINNING [0])
                                 drop-last)
               :output-bpes bpes
               :tokens      tokens})
            (map read-string random))))

(def get-count-filename
  (comp str
        :length))

(defn make-get-filename-content
  [dataset split]
  (fn [m]
    [(get-dataset-path dataset split (get-count-filename m))
     (str (generate-string m) "\n")]))

(defn spit-dataset
  [dataset coll & more]
  (run! (partial apply appending-spit-parents)
        (if (= dataset "simple")
          (s/select [s/ALL s/ALL]
                    (map (partial aid/funcall map)
                         (map (partial make-get-filename-content dataset)
                              ["validation" "training"])
                         (split-at (first more) coll)))
          (map (make-get-filename-content dataset "validation") coll))))

(defn split-dataset
  [dataset & more]
  (with-open [random-file (->> "random.txt"
                               (get-dataset-path dataset)
                               io/reader)]
    (with-open [bpe-file (->> "bpe.txt"
                              (get-dataset-path dataset)
                              io/reader)]
      (apply spit-dataset
             dataset
             (structure {:bpe    (line-seq bpe-file)
                         :index  (->> "index.edn"
                                      (get-dataset-path dataset)
                                      slurp
                                      read-string)
                         :random (line-seq random-file)})
             more))))

(def get-source-target
  (juxt identity
        (comp (partial (aid/flip join-paths) "sorted.txt")
              fs/parent)))

(defn get-source-targets*
  [dataset split]
  (->> #"\d+"
       (fs/find-files (get-dataset-path dataset split))
       (sort-by (comp read-string
                      fs/name))
       (map get-source-target)))

(defn get-source-targets
  [dataset]
  (if (= dataset "simple")
    (mapcat (partial get-source-targets* dataset) ["training" "validation"])
    (get-source-targets* dataset "validation")))

(def append-file
  (aid/build appending-spit-parents
             second
             (comp slurp
                   first)))

;This definition is slower and more memory efficient.
;(def append-file
;  (comp (partial apply aid/funcall)
;        (partial interleave [command/cat ">>"])))

(def sort-by-length
  (comp (partial run! append-file)
        get-source-targets))

(defn mung
  [dataset & more]
  (aid/mlet [_ (if (= dataset "simple")
                 (aid/mlet [_ (extract)]
                           (either/right (combine)))
                 (either/right ""))
             _ (parse dataset)
             _ (either/right (split-sentences dataset))
             _ (randomize dataset)
             _ (either/right (get-text dataset))
             _ (learn-bpe dataset)
             _ (apply-bpe dataset)]
            (build-vocabulary dataset)
            (apply split-dataset dataset more)
            (sort-by-length dataset)))
