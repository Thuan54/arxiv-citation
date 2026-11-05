import tempfile
import unittest
from pathlib import Path

from hierarchy import (
    _split_sentences,
    build_publication,
    find_document_root,
    gather_compilation_source,
    parse_latex_source,
)


class HierarchyTest(unittest.TestCase):
    def test_expands_only_reachable_tex_and_ignores_comments(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "main.tex").write_text(
                r"""
\documentclass{article}
\begin{document}
\section{Intro}
\input{chapters/one}
\includeonly{chapters/one}
\include{chapters/unused_include}
\end{document}
""",
                encoding="utf-8",
            )
            (root / "chapters").mkdir()
            (root / "chapters" / "one.tex").write_text(
                "Included content.", encoding="utf-8"
            )
            (root / "unused.tex").write_text(
                "This should not appear.", encoding="utf-8"
            )
            (root / "chapters" / "unused_include.tex").write_text(
                "The excluded include should not appear.", encoding="utf-8"
            )

            main = find_document_root(root)
            source = gather_compilation_source(main)
            parsed = parse_latex_source(source, "v1")

            self.assertIn("Included content.", source)
            self.assertNotIn("This should not appear.", source)
            self.assertNotIn("The excluded include should not appear.", source)
            self.assertEqual(parsed["children"][0]["content"], "Intro")

    def test_builds_sections_sentences_lists_formula_and_figure(self):
        source = r"""
\documentclass{article}
\begin{document}
\section{Introduction}
This is sentence one. This is sentence two.
\begin{itemize}
\item First point.
\item Second point.
\end{itemize}
\begin{equation}
x = \frac{y}{2} + 1
\end{equation}
$$
z = 2
$$
\begin{figure}
\includegraphics{plot.png}
\caption{A useful plot.}
\end{figure}
\section{References}
This bibliography prose must not be included.
\section*{References and Notes}
This second bibliography block must not be included either.
\begin{thebibliography}{1}
\bibitem{key} This entry must not become hierarchy content.
\end{thebibliography}
\printbibliography
\end{document}
"""
        tree = parse_latex_source(source, "v1")
        kinds = []
        contents = []

        def visit(node):
            kinds.append(node["kind"])
            contents.append(node["content"])
            for child in node["children"]:
                visit(child)

        visit(tree)
        self.assertIn("sentence", kinds)
        self.assertIn("list", kinds)
        self.assertEqual(kinds.count("list_item"), 2)
        self.assertEqual(kinds.count("formula"), 2)
        self.assertIn("figure", kinds)
        self.assertFalse(any(
            "bibliography" in content.lower()
            or "bibliography block" in content.lower()
            or "entry must not" in content.lower()
            for content in contents
        ))
        formula = next(
            node["content"]
            for node in tree["children"][0]["children"]
            if node["kind"] == "formula"
        )
        self.assertIn(r"\frac{y}{2}", formula)

    def test_sentence_split_protects_abbreviations_urls_and_math(self):
        sentences = _split_sentences(
            r"We use i.e. $x.y$ and https://example.org/a.b. "
            r"Fig. 1 shows the result. It works."
        )
        self.assertEqual(len(sentences), 3)
        self.assertIn("$x.y$", sentences[0])
        self.assertIn("https://example.org/a.b.", sentences[0])
        self.assertTrue(sentences[1].startswith("Fig. 1"))
        self.assertTrue(sentences[2].startswith("It works"))

    def test_deduplicates_content_across_versions_and_tracks_edges(self):
        first = r"""
\begin{document}
\section{Intro}
We study this problem.
\end{document}
"""
        second = r"""
\begin{document}
\section{Background}
We study this problem.
\end{document}
"""
        elements, hierarchy, versions = build_publication(
            "2401.00001",
            [("v1", first), ("v2", second), ("v3", None)],
        )
        shared = [
            row for row in elements
            if row["kind"] == "sentence"
            and row["content"] == "We study this problem."
        ]
        self.assertEqual(len(shared), 1)
        self.assertEqual(
            sum(row["element_id"] == shared[0]["element_id"] for row in hierarchy),
            2,
        )
        self.assertEqual([row["status"] for row in versions], [
            "parsed", "parsed", "empty",
        ])
        self.assertEqual(versions[2]["element_count"], 0)

    def test_preserves_citations_without_extracting_bibliography(self):
        source = r"""
\begin{document}
\section{Intro}
We build on \cite{citationkey}.
\bibliography{references}
\end{document}
"""
        elements, hierarchy, versions = build_publication(
            "2401.00002", [("v1", source)]
        )
        self.assertEqual(len(versions), 1)
        self.assertTrue(hierarchy)
        self.assertTrue(any(
            r"\cite{citationkey}" in row["content"]
            for row in elements
        ))
        self.assertFalse(any("BibTeX" in row["kind"] for row in elements))


if __name__ == "__main__":
    unittest.main()
