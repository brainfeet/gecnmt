(ns gecnmt.prepare
  (:require [gecnmt.command :as command]
            [gecnmt.helpers :as helpers]))

(def get-dataset-path
  (partial helpers/join-paths "resources/dataset"))

(def extract
  (partial command/python "bin/WikiExtractor.py"
           "--json"
           "-o"
           (get-dataset-path "simple/extracted")
           (get-dataset-path "simple/original.xml")))
