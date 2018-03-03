(ns gecnmt.helpers
  (:require [clojure.java.io :as io]))

(def join-paths
  (comp str
        io/file))
