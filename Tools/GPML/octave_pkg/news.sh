#!/bin/bash

sed -n  -e '/NEW/,/\n\n/p' README.md
