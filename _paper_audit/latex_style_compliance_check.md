# LaTeX Style & Template Compliance Check

This document provides a detailed compliance audit of the LaTeX source drafts in the project repository against the target **Springer LNCS** template format for the **FISAT 2026** conference.

---

## 1. Current Status & Style Classification
The repository contains two draft manuscripts:
1. `paper/main.tex` (Explainable PCA paper)
2. `XPCA-1.tex` (Coordinate Collapse paper)

Both files are currently formatted using the **IEEEtran** conference template:
$$\text{Document Class: } \texttt{\backslash documentclass[conference]\{IEEEtran\}}$$
This is a double-column, US letter layout. The target conference (FISAT 2026) requires the **Springer LNCS** style, which is a single-column, A5-like layout running under the `llncs` document class:
$$\text{Target Class: } \texttt{\backslash documentclass\{llncs\}}$$

---

## 2. Key Template Discrepancies & Required Overrides

### A. Document Class Declaration
* **Current:** `\documentclass[conference]{IEEEtran}`
* **Springer LNCS:** `\documentclass[runningheads]{llncs}`
* *Action:* Replace the class header. The `runningheads` option is highly recommended by Springer to generate page headers.

### B. Title, Authors, and Affiliations
IEEEtran and LNCS structure author blocks in completely different ways:
* **Current (IEEEtran):**
  Uses `\IEEEauthorblockN{...}` and `\IEEEauthorblockA{...}` inside a multi-column block.
* **Springer LNCS:**
  Uses a structured list of authors, running author headers, and numbered institutes:
  ```latex
  \title{Title of the Paper}
  \titlerunning{Abbreviated Running Title}
  
  \author{Nguyen Duc Bao Lam\inst{1} \and
  Nguyen Cong Thinh\inst{1} \and
  Tran Trung Tin\inst{1} \and
  Vo Hieu Thang\inst{1} \and
  Huy Nguyen Xuan\inst{1}\thanks{Corresponding author.} \and
  Trung Nguyen Quoc\inst{1} \and
  Tai Nguyen Trong\inst{1}}
  
  \authorrunning{N. D. B. Lam et al.}
  
  \institute{FPT University, Hanoi, Vietnam\\
  \email{{lam01662052827, nguyencongthinh17122006, trungtin1218, thanhtuan21062000}@gmail.com}\\
  \email{{huynx23, trungnq46, taint51}@fpt.edu.vn}}
  ```

### C. Abstract and Keywords
* **Current (IEEEtran):**
  Keywords are defined outside the abstract environment using:
  ```latex
  \begin{IEEEkeywords}
  keyword1, keyword2.
  \end{IEEEkeywords}
  ```
* **Springer LNCS:**
  Keywords are placed inside the `abstract` environment using the `\keywords{...}` command:
  ```latex
  \begin{abstract}
  This is the abstract text.
  \keywords{keyword1 \and keyword2 \and keyword3}
  \end{abstract}
  ```

### D. Theorem and Definition Environments
* **Current (XPCA-1.tex lines 18-22):**
  Uses `\newtheorem{theorem}{Theorem}` to manually define environments.
* **Springer LNCS:**
  The `llncs` class already defines standard theorem-like environments internally (e.g. `theorem`, `proposition`, `corollary`, `definition`, `remark`, `proof`).
* *Action:* Remove manual `\newtheorem` declarations in `XPCA-1.tex`. Otherwise, the LaTeX compiler will throw errors:
  $$\texttt{LaTeX Error: Command \backslash theorem already defined.}$$

### E. Bibliography Style
* **Current (IEEEtran):**
  Uses `\bibliographystyle{IEEEtran}` or standard IEEE bibliography items.
* **Springer LNCS:**
  Requires the Springer numbered style:
  ```latex
  \bibliographystyle{splncs04}
  \bibliography{references}
  ```

### F. Page Balance & Column Layout
* **Current (IEEEtran):**
  Uses the `balance` package and the `\balance` command in the references section to balance the heights of the two columns on the final page.
* **Springer LNCS:**
  Is a single-column layout. The `\balance` command has no function and must be removed to avoid compile warnings/errors.

---

## 3. Step-by-Step Conversion Guidelines
For a reviewer or AI assistant to compile these files under Springer LNCS format, follow this procedure:

1. **Obtain the template class:** Copy the `llncs.cls` and `splncs04.bst` files into the compilation directory.
2. **Replace the Document Class:**
   ```diff
   -\documentclass[conference]{IEEEtran}
   +\documentclass[runningheads]{llncs}
   ```
3. **Remove Redundant Style Packages:** Remove any references to column balancing or manual geometry overrides.
4. **Reformat Title and Author Blocks:** Replace the `\IEEEauthorblock` macros with the `\author`, `\authorrunning`, and `\institute` block structure shown above.
5. **Nest Keywords inside the Abstract:** Remove the `IEEEkeywords` environment and add `\keywords{...}` inside the abstract.
6. **Clean up Theorem Declarations:** Delete the `\newtheorem` declarations from the preamble.
7. **Change Bibliography Class:** Change the bibliography style to `splncs04` and verify that the items comply with the Springer citation style.
