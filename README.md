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

Two executable are install with this module, see help

```bash
biaxpp -h
```
```bash
bxconv -h
```

# Limitations
  - TBD

# License

TBD
