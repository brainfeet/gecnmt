(ns gecnmt.rsync
  (:require [gecnmt.command :as command]))

(defn rsync
  [{uri :uri}]
  (command/rsync "-azP"
                 "--include=resources/hyperparameter/*"
                 "--exclude=.idea"
                 ;.gitignore in the project root seems to be used as a filter by default
                 "--filter=':- /python/.gitignore'"
                 (System/getProperty "user.dir")
                 uri))
