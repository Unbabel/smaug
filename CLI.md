## Command Line Interface

This document describes the command line interface provided by this package. There are three types of operations: Transforms, Validations and Utilities.

Transforms take as input a sentence and produce one or multiple perturbed sentences.

Validations receive an original sentence and a perturbed sentence and verify if the pertubed sentece complies with some requirement.

> ***NOTE***: Each transform defines some *default* validations to be executed, which can be disabled with the `no-post-run` flag.

Utilities offer functions to perform common operations, such as reading and writing files.

## Configuration File Specification

The cli tool can be used with a `yaml` configuration file as follows:

```
augment --cfg <path_to_config_file>
```

An example of a configuration file is:

```yaml
pipeline:
- cmd: io-read-csv
  path: <path to input file>
- cmd: transf-neg
- cmd: transf-ins-text
  validations:
  - cmd: val-keep-geq-edit-dist
    distance: 8
    level: word
- cmd: val-rm-pattern
  pattern: hello-world
- cmd: io-write-json
  path: <path to output file>
seed: 42
no-post-run: False
```

The first pipeline section is mandatory and specifies a list with all the commands to be executed. After that section, other cli arguments can be specified (such as `seed` in this example). The arguments are the same as in the cli command, but without the `--` in the beginning. Boolean flags also do not have `--` and can have the value True of False.


Inside the pipeline section, each command is identified with `cmd: <command name>`. The remaining tags in the command entry are the arguments for the command. 

Inside transforms, a special `validations` tag can be used to specify validations for the command only. Validations for all previous transforms can be specified as a regular command in the pipeline. In the above exaple `val-keep-geq-edit-dist` is only applied to `transf-ins-text` but `val-rm-pattern` is applied to `transf-neg` and `transf-ins-text`.

## Transforms

### transf-swp-ne

Detects a single named entity with a [Stanza NER model](https://stanfordnlp.github.io/stanza/available_models.html#available-ner-models) and swaps it for text generated with [Google's mT5](https://arxiv.org/abs/2010.11934).

### transf-swp-num

Detects a single number with RegEx and swaps it for text generated with [Google's mT5](https://arxiv.org/abs/2010.11934).

### transf-neg

Negates an english sentence using [PolyJuice](https://arxiv.org/abs/2101.00288) conditioned for negation.

### transf-ins-text

Insert random text in multiple places using [Google's mT5](https://arxiv.org/abs/2010.11934) model.

### transf-del-punct-span

Removes a single span between two punctuation symbols `.,?!`.

The following table details the available CLI commands:

## Validations

### val-rm-equal

Verifies if the perturbed sentence is different from the original sentence with string comparison. Useful if the transform may return the original sentence.

### val-rm-pattern

Verifies if the perturbed sentence does not have a specific regular expression. Useful with language models that may leave special tokens behind.

### val-keep-contradiction

Verifies if the perturbed sentence contradicts the original sentence. Relies on a [RoBERTa](https://arxiv.org/abs/1907.11692) model trained for mnli.

### val-keep-eq-ne

Verifies if the perturbed and original sentences have the same number of named entities using a [Stanza NER model](https://stanfordnlp.github.io/stanza/available_models.html#available-ner-models) to detect them.

### val-keep-eq-num

Verifies if the perturbed and original sentences have the same number of numbers using RegEx to detect them.

### val-keep-leq-char-ins

Verifies if the perturbed sentence has a number of specific character insertions below a threshold, when compared to the original.

### val-keep-geq-edit-dist

Verifies if the perturbed and original sentences an [minimum edit distance](https://web.stanford.edu/class/cs124/lec/med.pdf) above a threshold.

## Utilities

### io-read-lines

Reads sentences from a text file, where each line is a sentence.

### io-read-csv

Reads the sentences from a csv file. Each line of the file has the sentence to perturb and the sentence language in the format \<lang code\>,\<sentence\>.

### io-write-json

Writes the perturbed sentences in a human-readable JSON format. Each input sentence has a respective output JSON object (in the order of the input). Each JSON object has the original sentence, a dictionary with the perturbations indentified by the transform name and metadata for each transform (also identified by the transform name).