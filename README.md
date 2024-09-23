---
title: HOME PAGE
filename: README.md
description: |
  This is the README page for providing general information for using this code.
  More detailed information is provided on other pages through links
---

Biaxial Data Analysis
==================================================

**Authors:** *Will Zhang*

## Pages
> * [Home](README.md)


# About / Synopsis

* For the collaboration project with Alexey Kamenskiy et al.
* Project status: pilot testing
* Please cite

Zhang, W., Feng, Y., Lee, C., Billiar, K. L., and Sacks, M. S. (June 1, 2015). "A Generalized Method for the Analysis of Planar Biaxial Mechanical Data Using Tethered Testing Configurations." ASME. J Biomech Eng. June 2015; 137(6): 064501. https://doi.org/10.1115/1.4029266

## Features

* Python interface
* 2D Biax data processing

## Table of contents

> * [About / Synopsis](#about--synopsis)
>   * [Features](#features)
>   * [Table of contents](#table-of-contents)
> * [Installation](#installation)
>   * [Requirements](#requirements)
> * [Usage](#usage)
> * [Limitations](#limitations)
> * [License](#license)

# Installation

Python 3.11 or greater is required.

## Compiling Cython module

Make a virtual environment
```bash
python3 -m venv .venv
```
Activate environment
```bash
source .venv/bin/activate
```
```Powershell
.venv/Script/activate
```
Install module
```bash
python3 -m pip install -e sacksbiax
```


# Usage

Two executable are install with this module, to see help,

```bash
biaxpp -h
```
```bash
bxconv -h
```
To post process raw data from Dr. Sacks' device using old method
```bash
biaxpp "50001__N 001__pPA L" --method PK1
```
To post process raw data from Dr. Sacks' device using new method
```bash
biaxpp "50001__N 001__pPA L" --method CAUCHY
```

To update existing data using the new method
```bash
bxconv "00613__N 309__pSFA R/01 - All data.xlsx" --method CAUCHY
```


# Limitations
  - TBD

# License

TBD
