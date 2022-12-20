# ShapeNet Utilities

A utility repository for creating interoperability between ShapeNet-based datasets such as:

- ShapeNetCore V1
- ShapeNetCore V2
- ShapeNet Sem
- PartNet
- ACRONYM
- ShapeNet Part

Even though all the above datasets share the same set of base meshes, meshes within each of the datasets are transformed differently from the base meshes. Therefore, if you wish to incorporate labels from one dataset into another, a significant amount of effort is needed to align the datasets. This repository aims to help with the process.

## Prerequisites

Please download the latest version of ShapeNetCore dataset from [this website](https://shapenet.org/).

## Aligning ShapeNet Part dataset

To align the ShapeNet Part dataset with ShapeNetCore V2, simply run `python main.py`.