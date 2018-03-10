(ns gecnmt.core
  (:require [gecnmt.mung :as mung]
            [gecnmt.rsync :as rsync]))

(def hyperparameter
  (-> "resources/hyperparameter/hyperparameter.edn"
      slurp
      read-string))

(defn -main
  [command]
  (println (({"mung"  mung/mung
              "rsync" rsync/rsync}
              command)
             hyperparameter))
  ;mung/mung doesn't exit immediately
  (shutdown-agents))
