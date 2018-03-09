# gecnmt

FIXME: description

## Installation

### Clojure
1. Download the lein script (or on Windows lein.bat)
2. Place it on your $PATH where your shell can find it (eg. ~/bin)
3. Set it to be executable (chmod a+x ~/bin/lein)
4. Run it (lein) and it will download the self-install package

https://leiningen.org/

### Python
```
conda install -f python/gpu.yml --force
source activate gecnmt
python -m spacy download en
```

## Usage

```
source activate gecnmt
lein run munge
cd python
export PYTHONPATH=$(pwd)
python gecnmt/train.py
```

## Options

resources/hyperperameters/hyperperameter.edn
python/hyperperameters/hyperperameter.json

## Examples

...

### Bugs

...

### Any Other Sections
### That You Think
### Might be Useful

## License

Copyright © 2018 FIXME

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
