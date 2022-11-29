# SMAUG: Sentence-level Multilingual AUGmentation

`smaug` is a package for multilingual data augmentation. It offers transformations focused on changing specific aspects of sentences, such as Named Entities, Numbers, etc.

# Getting Started

To run a simple pipeline with all transforms and default validations, first create the following `yaml` file:

```yaml
pipeline:
- cmd: io-read-lines
  path: <path to input file with single sentence per line>
  lang: <two letter language code for the input sentences>
- cmd: transf-swp-ne
- cmd: transf-swp-num
- cmd: transf-neg
- cmd: transf-ins-text
- cmd: transf-del-punct-span
- cmd: io-write-json
  path: <path to output file>
# Remove this line for no seed
seed: <seed for the pipeline>
```

The run the following command:

```shell
augment --cfg <path_to_config_file>
```

# Usage

The `smaug` package can be used as a command line interface (CLI) or by directly importing and calling the package Python API. To use `smaug`, first install it by following these [instructions](#install).

## Command Line Interface

The CLI offers a way to read, transform, validate and write perturbed sentences to files. For more information, see the [full details](CLI.md).

### Configuration File

The easiest way to run `smaug` is through a configuration file (see the [full specification](CLI.md#configuration-file-specification)) that specifies and entire pipeline (as shown in the [Getting Started](#getting-started) section), using the following command:

```shell
augment --cfg <path_to_config_file>
```

### Single transform

As an alternative, you can use the command line to directly specify the pipeline to apply. To apply a single transform to a set of sentences, execute the following command:

```shell
augment io-read-lines -p <input_file> -l <input_lang_code> <transf_name> io-write-json -p <output_file>
```

> `<transf_name>` is the name of the transform to apply (see this [section](OPERATIONS.md#transforms) for a list of available transforms).
>
> `<input_file>` is a text file with one sentence per line.
>
> `<input_lang_code>` is a two character language code for the input sentences.
>
> `<output_file>` is a json file to be created with the transformed sentences.

### Multiple Transforms

To apply multiple transforms, just specify them in arbitrary order between the read and write operations:

``` shell
augment io-read-lines -p <input_file> -l <input_lang_code> <transf_name_1> <transf_name_2> ... io-write-json -p <output_file>
```

### Multiple Inputs

To read from multiple input files, also specify them in arbitrary order:

```shell
augment io-read-lines -p <input_file_1> -l <input_lang_code_1> read-lines -p <input_file_2> -l <input_lang_code_2> ... <transf_name_1> <transf_name_2> ... io-write-json -p <output_file>
```

You can further have multiple languages in a given file by having each line with the structure \<lang code\>,\<sentence\> and using the following command:

```shell
augment io-read-csv -p <input_file> <transf_name_1> <transf_name_2> ... io-write-json -p <output_file>
```

# Install

To install this package, execute the following steps:

* Install the [poetry](https://python-poetry.org/docs/#installation) tool for dependency management.

* Clone this git repository and install the project.

```
git clone https://github.com/Unbabel/smaug.git
cd smaug
poetry install
```