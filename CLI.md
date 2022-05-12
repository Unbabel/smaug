## Command Line Interface

This document describes the command line interface provided by this package. There are three types of operations: Transforms, Validations and Utilities.

Transforms take as input a sentence and produce one or multiple perturbed sentences.

Validations receive an original sentence and a perturbed sentence and verify if the pertubed sentece complies with some requirement.

Utilities offer functions to perform common operations, such as reading and writing files.

The following table details the available CLI commands:

<table>
	<tr>
        <th></th>
	    <th>Name</th>
	    <th>Command</th>
	    <th>Description</th>  
	</tr >
	<tr >
	    <td rowspan="4">Transform</td>
		<td>Delete Span between Punct</td>
	    <td nowrap="nowrap"><code>transf-del-punct-span</code></td>
        <td>Removes a single span between two punctuation symbols (<code>.,?!</code>).</td>
	</tr>
	<tr>
        <td>Negate</td>
	    <td nowrap="nowrap"><code>transf-neg</code></td>
        <td>Negates an english sentence using <a href="https://arxiv.org/abs/2101.00288">PolyJuice</a> conditioned for negation.</td>
	</tr>
	<tr>
	    <td>Swap Named Entity</td>
	    <td><code>transf-swp-ne</code></td>
        <td>Detects a single named entity with a <a href="https://stanfordnlp.github.io/stanza/available_models.html#available-ner-models">Stanza model</a> and swaps it for text generated with <a href="https://arxiv.org/abs/2010.11934">Google's mT5</a>.</td>
	</tr>
	<tr>
	    <td>Swap Number</td>
	    <td nowrap="nowrap"><code>transf-swp-num</code></td>
        <td>Detects a single number with RegEx and swaps it for text generated with <a href="https://arxiv.org/abs/2010.11934">Google's mT5</a>.</td>
	</tr>
	<tr>
	    <td rowspan="6">Validation</td>
	    <td>Keep Contradiction</td>
        <td nowrap="nowrap"><code>val-keep-contradiction</code></td>
        <td>Verifies if the perturbed sentence contradicts the original sentence. Relies on a <a href="https://arxiv.org/abs/1907.11692">RoBERTa</a> model trained for mnli.</td>
	</tr>
	<tr>
	    <td>Keep Equal Numbers Count</td>
	    <td nowrap="nowrap"><code>val-keep-eq-num</code></td>
	    <td>Verifies if the perturbed and original sentences have the same number of numbers using RegEx to detect them.</td>
	</tr>
	<tr>
	    <td>Keep Equal Named Entities Count</td>
	    <td nowrap="nowrap"><code>val-keep-eq-ne</code></td>
	    <td>Verifies if the perturbed and original sentences have the same number of named entities using a <a href="https://stanfordnlp.github.io/stanza/available_models.html#available-ner-models">Stanza model</a> to detect them.</td>
	</tr>
	<tr>
	    <td>Keep Greater or Equal Edit Distance</td>
	    <td nowrap="nowrap"><code>val-keep-geq-edit-dist</code></td>
	    <td>Verifies if the perturbed and original sentences an <a href="https://web.stanford.edu/class/cs124/lec/med.pdf">minimum edit distance</a> above a threshold.</td>
	</tr>
	<tr>
	    <td>Remove Equal Sentences</td>
	    <td nowrap="nowrap"><code>val-rm-equal</code></td>
	    <td>Verifies if the perturbed sentence is different from the original sentence with string comparison. Useful if the transform may return the original sentence.</td>
	</tr>
	<tr>
	    <td>Remove a Pattern</td>
	    <td nowrap="nowrap"><code>val-rm-pattern</code></td>
	    <td>Verifies if the perturbed sentence does not have a specific regular expression. Useful with language models that may leave special tokens behind.</td>
	</tr>
	<tr>
        <td rowspan="5">Utility</td>
	    <td>Read Lines</td>
	    <td nowrap="nowrap"><code>read-lines</code></td>
	    <td>Reads sentences from a text file, where each line is a sentence.</td>
	</tr>
    <tr>
	    <td>Read CSV</td>
	    <td nowrap="nowrap"><code>read-csv</code></td>
	    <td>Reads a csv file with multiple language pairs, storing all columns. The original sentence will be associated to the "ref" column.</td>
	</tr>
    <tr>
	    <td>Write JSON </td>
	    <td nowrap="nowrap"><code>write-json</code></td>
	    <td>Writes the perturbed sentences in a human-readable JSON format.</td>
	</tr>
</table>
