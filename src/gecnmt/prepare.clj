(ns gecnmt.prepare
  (:require [clojure.java.shell :as sh]
            [clojure.string :as str]
            [aid.core :as aid]
            [cheshire.core :refer :all]
            [me.raynes.fs :as fs]
            [gecnmt.command :as command]
            [gecnmt.helpers :as helpers]))

(def get-dataset-path
  (partial helpers/join-paths "resources/dataset"))

(def extract
  (partial command/python "bin/WikiExtractor.py"
           "--json"
           "-o"
           (get-dataset-path "simple/extracted")
           (get-dataset-path "simple/original.xml")))

(def parse-keywordize
  (partial (aid/flip parse-string) true))

(def parse-extracted
  (comp (partial map parse-keywordize)
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

(def spit-isolated
  (aid/build spit-parents
             (comp (partial (aid/flip str) ".txt")
                   (partial get-dataset-path "simple/isolated")
                   :id)
             :text))

(def isolate
  (comp (partial run! spit-isolated)
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
        (partial (aid/flip get-dataset-path) "isolated")))
