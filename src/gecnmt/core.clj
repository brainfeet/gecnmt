(ns gecnmt.core
  (:require [gecnmt.mung :as mung]))

(defn -main
  [command]
  (println (({"mung" mung/mung} command)))
  ;mung/mung doesn't exit immediately
  (shutdown-agents))
