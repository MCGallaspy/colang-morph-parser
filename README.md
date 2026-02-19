# What is this?

An app for creating a morphological parser of a language using an active learning paradigm.

# What can you do with it?

You can do the following things. The idea is to initialize a model, and then switch between training/evaluating it and labeling data interactively.

## Initialize a model

Initialize a morphological parsing model. You must specify what parts of the morphology you want to model and indicate it at this point. You don't have to model all morphology in a language; you can model only the aspects of morphology that are of interest to you. However the choice of what parts of the language to model are "locked in" to a model after initialization.

## Train and evaluate model

Train and evaluate a model on labeled data.

## Label data

Use a model to estimate which data are the most difficult (in technical terms, which predictions have the highest entropy). Then manually label that data.