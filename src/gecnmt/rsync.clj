(ns gecnmt.rsync
  (:require [gecnmt.command :as command]))

(defn rsync
  [{target :target}]
  (command/rsync "-azP"
                 "--exclude=.idea"
                 ;.gitignore in the project root seems to be used as a filter by default
                 "--filter=':- /python/.gitignore'"
                 (System/getProperty "user.dir")
                 target))
