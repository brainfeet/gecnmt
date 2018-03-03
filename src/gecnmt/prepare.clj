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

(def split-sentences*
  (comp (partial map flatten)
        (partial partition 2)
        (partial partition-by :is_sent_start)
        (partial s/setval* [s/FIRST :is_sent_start] true)
        parse-keywordize))

(defn spit-line
  [f content]
  (spit f (str content "\n") :append true))

(defn split-sentences
  [dataset]
  (with-open [f (io/reader ((partial (aid/flip get-dataset-path) "parsed.txt") dataset))]
    (run! (partial spit-line ((partial (aid/flip get-dataset-path) "split.txt") dataset))
          (map vec (mapcat split-sentences*
                           (line-seq f))))))
