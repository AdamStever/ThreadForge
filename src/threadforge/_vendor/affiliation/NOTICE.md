# Vendored: affiliation-metrics

This directory contains the **affiliation metrics** reference implementation,
kept **verbatim** (not modified). It computes the affiliation-based precision and
recall of Huet, Greff & Tartakovsky, *"Local Evaluation of Time Series Anomaly
Detection Algorithms"* (KDD 2022).

It is vendored — rather than reimplemented — deliberately: the metric's value is
exact fidelity to the published method, and its precision/recall are computed by
closed-form integration over "affiliation zones" (`_integral_interval.py`), where
a from-scratch rewrite would add transcription risk for no benefit. ThreadForge's
own metric code lives in `tab_scoring.py`, which calls into this package.

- **Original author:** Alexis Huet et al.
- **Upstream source:** https://github.com/ahstat/affiliation-metrics-py
  (also bundled in https://github.com/TheDatumOrg/VUS)
- **License:** MIT

```
MIT License

Copyright (c) 2022 Alexis Huet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
