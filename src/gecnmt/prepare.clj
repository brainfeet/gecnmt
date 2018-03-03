(ns gecnmt.prepare
  (:require [clojure.java.io :as io]
            [clojure.java.shell :as sh]
            [clojure.string :as str]
            [aid.core :as aid]
            [cheshire.core :refer :all]
            [me.raynes.fs :as fs]
            [gecnmt.command :as command]
            [gecnmt.helpers :as helpers]
            [com.rpl.specter :as s]))

(def get-dataset-path
  (partial helpers/join-paths "resources/dataset"))

(def extract
  (partial command/python "bin/WikiExtractor.py"
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

(defn spit-combined
  [content]
  (spit (get-dataset-path "simple/combined.txt")
        content
        :append
        true))

(def combine
  (comp (partial run! spit-combined)
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


(defn appending-spit
  [f content]
  (spit f content :append true))

(defn split-sentences
  [dataset]
  (with-open [f (->> "parsed.txt"
                     (get-dataset-path dataset)
                     io/reader)]
    (->> f
         line-seq
         (mapcat split-sentences*)
         (run! (partial appending-spit (get-dataset-path dataset
                                                         "split.txt"))))))
(def randomize
  (aid/build (partial command/shuf "-o")
             (partial (aid/flip get-dataset-path) "random.txt")
             (partial (aid/flip get-dataset-path) "split.txt")))
