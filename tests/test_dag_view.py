"""Tests for the box-and-arrow DAG renderer (care.runtime.dag_view)."""

from __future__ import annotations

from rich.text import Text

from care.runtime.dag_view import (
    render_dag_boxes,
    render_dag_diff,
    render_dag_styled,
)


def _joined(payload, **kw) -> str:
    return "\n".join(render_dag_boxes(payload, **kw))


def _styles(lines) -> set[str]:
    """Collect every Rich style string present across a list of Text lines —
    both whole-line styles (cycle/legend rows) and per-span styles (the
    canvas glyphs)."""
    out: set[str] = set()
    for ln in lines:
        if ln.style:
            out.add(str(ln.style))
        for span in ln.spans:
            if span.style:
                out.add(str(span.style))
    return out


class TestEmpty:
    def test_empty_dict_returns_empty_list(self):
        assert render_dag_boxes({}) == []

    def test_none_returns_empty_list(self):
        assert render_dag_boxes(None) == []

    def test_no_recognised_nodes_returns_empty(self):
        assert render_dag_boxes({"foo": "bar"}) == []


class TestLinearChain:
    STEPS = {
        "steps": [
            {"number": 1, "title": "Analyse query", "type": "llm",
             "dependencies": []},
            {"number": 2, "title": "Fetch data", "type": "tool",
             "dependencies": [1]},
            {"number": 3, "title": "Summarise", "type": "llm",
             "dependencies": [2]},
        ]
    }

    def test_each_step_gets_a_box(self):
        out = _joined(self.STEPS)
        assert "Analyse query (AI)" in out
        assert "Fetch data (Tool)" in out
        assert "Summarise (AI)" in out

    def test_boxes_drawn_with_borders(self):
        out = _joined(self.STEPS)
        assert "┌" in out and "┐" in out
        assert "└" in out and "┘" in out

    def test_arrow_spine_between_boxes(self):
        out = _joined(self.STEPS)
        # Two arrows for three boxes.
        assert out.count("▼") == 2
        assert "│" in out

    def test_linear_deps_are_not_annotated(self):
        # Each step depends only on the box directly above it, so the
        # down-arrow already conveys the edge — no `◀` clutter.
        assert "◀" not in _joined(self.STEPS)

    def test_topological_order_preserved(self):
        out = _joined(self.STEPS)
        assert out.index("Analyse query") < out.index("Fetch data")
        assert out.index("Fetch data") < out.index("Summarise")


class TestDiamond:
    """A fork/join diamond: 1 → {2, 3} → 4."""

    GRAPH = {
        "nodes": [
            {"id": 1, "name": "root", "type": "llm"},
            {"id": 2, "name": "left", "type": "tool"},
            {"id": 3, "name": "right", "type": "tool"},
            {"id": 4, "name": "merge", "type": "llm"},
        ],
        "edges": [
            {"from": 1, "to": 2},
            {"from": 1, "to": 3},
            {"from": 2, "to": 4},
            {"from": 3, "to": 4},
        ],
    }

    def test_all_nodes_rendered_once(self):
        out = _joined(self.GRAPH)
        assert out.count("root (AI)") == 1
        assert out.count("merge (AI)") == 1

    def test_parallel_steps_share_a_row(self):
        # The two parallel branches sit on the same line (side by side),
        # not stacked vertically.
        lines = render_dag_boxes(self.GRAPH)
        row = next(ln for ln in lines if "left" in ln)
        assert "right" in row

    def test_all_edges_drawn_no_annotation(self):
        # Every edge spans exactly one layer, so all are drawn as lines
        # — nothing falls back to a ◀ annotation.
        assert "◀" not in _joined(self.GRAPH)

    def test_fork_and_join_junctions_present(self):
        out = _joined(self.GRAPH)
        # Fork splitter (┴ feeds two children) and join (┬ merges two
        # parents) both appear.
        assert "┴" in out
        assert "┬" in out

    def test_topo_respects_edges(self):
        out = _joined(self.GRAPH)
        assert out.index("root") < out.index("merge")


class TestParallelColumns:
    def test_wide_fanout_renders_columns_on_one_row(self):
        out = render_dag_boxes({
            "nodes": [
                {"number": 1, "title": "split", "type": "tool"},
                {"number": 2, "title": "A", "type": "llm"},
                {"number": 3, "title": "B", "type": "llm"},
                {"number": 4, "title": "C", "type": "llm"},
            ],
            "edges": [
                {"from": 1, "to": 2},
                {"from": 1, "to": 3},
                {"from": 1, "to": 4},
            ],
        })
        # All three branches land on a single row.
        row = next(ln for ln in out if "2 · A" in ln)
        assert "3 · B" in row and "4 · C" in row
        # A fan-out crossing junction connects the splitter to 3 kids.
        assert "┼" in "\n".join(out)

    def test_independent_roots_share_a_row_without_edges(self):
        out = render_dag_boxes({"nodes": [1, 2, 3]})
        assert len(out) == 3  # all three boxes occupy one 3-line row
        joined = "\n".join(out)
        assert joined.count("┌") == 3
        assert "▼" not in joined


class TestSkipEdges:
    def test_multi_layer_edge_is_annotated_not_drawn(self):
        # 1→2→3→4 plus a direct 1→4 edge that spans three layers.
        out = _joined({
            "nodes": [
                {"number": 1, "title": "A", "type": "llm"},
                {"number": 2, "title": "B", "type": "llm"},
                {"number": 3, "title": "C", "type": "llm"},
                {"number": 4, "title": "D", "type": "llm"},
            ],
            "edges": [
                {"from": 1, "to": 2},
                {"from": 2, "to": 3},
                {"from": 3, "to": 4},
                {"from": 1, "to": 4},
            ],
        })
        # The long edge surfaces as an annotation on step 4.
        assert "◀ 1" in out
        # The chain itself stays a clean single column (3 spine arrows).
        assert out.count("▼") == 3


class TestPayloadShapes:
    def test_bare_list_of_step_dicts(self):
        out = _joined([
            {"number": 1, "name": "a", "type": "llm"},
            {"number": 2, "name": "b", "type": "tool", "dependencies": [1]},
        ])
        assert "a (AI)" in out
        assert "b (Tool)" in out

    def test_bare_scalar_nodes(self):
        out = _joined({"nodes": [1, 2, 3]})
        # No edges → three independent boxes, no arrows.
        assert out.count("┌") == 3
        assert "▼" not in out

    def test_edges_as_source_target(self):
        out = _joined({
            "nodes": [{"id": "a", "name": "a"}, {"id": "b", "name": "b"}],
            "edges": [{"source": "a", "target": "b"}],
        })
        assert out.index("a") < out.index("b")
        assert "▼" in out

    def test_edges_as_pairs(self):
        out = _joined({
            "nodes": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            "edges": [[1, 2]],
        })
        assert "▼" in out


class TestTypeLabels:
    def test_known_types_mapped(self):
        out = _joined({"nodes": [
            {"id": 1, "name": "x", "type": "llm"},
            {"id": 2, "name": "y", "type": "mcp"},
        ]})
        assert "(AI)" in out
        assert "(MCP)" in out

    def test_unknown_type_titlecased(self):
        out = _joined({"nodes": [{"id": 1, "name": "x", "type": "weird_kind"}]})
        assert "(Weird Kind)" in out

    def test_no_type_no_suffix(self):
        out = _joined({"nodes": [{"id": 1, "name": "plain"}]})
        assert "plain" in out
        assert "(" not in out.split("plain")[1].split("│")[0]


class TestCycle:
    def test_cycle_is_flagged_not_hung(self):
        out = _joined({"steps": [
            {"id": "a", "type": "llm", "deps": ["b"]},
            {"id": "b", "type": "tool", "deps": ["a"]},
        ]})
        assert "cycle detected" in out
        # Both nodes still rendered despite the cycle.
        assert "step a" in out and "step b" in out


class TestWidth:
    def test_long_label_truncated_with_ellipsis(self):
        out = _joined(
            {"nodes": [{"id": 1, "name": "x" * 200, "type": "llm"}]},
            max_width=20,
        )
        assert "…" in out
        # No rendered content line exceeds the box width budget.
        for line in out.splitlines():
            assert len(line) <= 20 + 4

    def test_uniform_box_width(self):
        out = render_dag_boxes({"nodes": [
            {"id": 1, "name": "short"},
            {"id": 2, "name": "a much longer label here", "deps": [1]},
        ]})
        tops = [ln for ln in out if ln.startswith("┌")]
        assert len(tops) == 2
        # Both boxes share one width so the spine lines up.
        assert len({len(t) for t in tops}) == 1


class TestCompactMode:
    WIDE = {
        "nodes": [
            {"number": 1, "title": "Split corpus", "type": "tool"},
            {"number": 2, "title": "Sentiment", "type": "llm"},
            {"number": 3, "title": "Topics", "type": "llm"},
            {"number": 4, "title": "Entities", "type": "llm"},
            {"number": 5, "title": "Merge report", "type": "llm"},
        ],
        "edges": [
            {"from": 1, "to": 2}, {"from": 1, "to": 3}, {"from": 1, "to": 4},
            {"from": 2, "to": 5}, {"from": 3, "to": 5}, {"from": 4, "to": 5},
        ],
    }

    def _split(self, lines):
        """Return (graph_lines, legend_lines) around the `steps:` marker."""
        idx = lines.index("steps:")
        return lines[:idx], lines[idx:]

    def test_wide_graph_triggers_compact(self):
        lines = render_dag_boxes(self.WIDE, max_graph_width=40)
        assert "steps:" in lines

    def test_boxes_hold_bare_numbers_not_labels(self):
        lines = render_dag_boxes(self.WIDE, max_graph_width=40)
        graph, _ = self._split(lines)
        graph_text = "\n".join(graph)
        # The number boxes are present...
        assert "│ 1 │" in graph_text
        assert "│ 5 │" in graph_text
        # ...but no label text leaks into the graph itself.
        assert "Sentiment" not in graph_text
        assert "Topics" not in graph_text

    def test_legend_lists_every_step(self):
        lines = render_dag_boxes(self.WIDE, max_graph_width=40)
        _, legend = self._split(lines)
        legend_text = "\n".join(legend)
        for n, name in [
            (1, "Split corpus"), (2, "Sentiment"), (3, "Topics"),
            (4, "Entities"), (5, "Merge report"),
        ]:
            assert f"{n} — {name}" in legend_text

    def test_legend_sorted_by_number(self):
        lines = render_dag_boxes(self.WIDE, max_graph_width=40)
        _, legend = self._split(lines)
        entries = [ln for ln in legend if "—" in ln]
        refs = [ln.split(" — ")[0] for ln in entries]
        assert refs == ["1", "2", "3", "4", "5"]

    def test_compact_graph_still_draws_edges(self):
        lines = render_dag_boxes(self.WIDE, max_graph_width=40)
        graph, _ = self._split(lines)
        graph_text = "\n".join(graph)
        # Fork + join junctions survive in the compact graph.
        assert "▼" in graph_text
        assert "┼" in graph_text

    def test_compact_graph_fits_width_budget(self):
        lines = render_dag_boxes(self.WIDE, max_graph_width=40)
        graph, _ = self._split(lines)
        # The number-box graph is dramatically narrower than the
        # full-label layout would have been.
        assert max(len(ln) for ln in graph) < 40

    def test_narrow_graph_stays_full_labelled(self):
        # A small graph under the budget keeps labels inside the boxes
        # and emits no legend.
        lines = render_dag_boxes({
            "steps": [
                {"number": 1, "title": "Ask", "type": "llm"},
                {"number": 2, "title": "Answer", "type": "llm",
                 "dependencies": [1]},
            ]
        })
        assert "steps:" not in lines
        assert any("Ask (AI)" in ln for ln in lines)

    def test_skip_dep_moves_to_legend(self):
        # 1→2→3→4 plus a long 1→4 edge, forced compact.
        lines = render_dag_boxes({
            "nodes": [
                {"number": 1, "title": "Aaaaaaaaaa", "type": "llm"},
                {"number": 2, "title": "Bbbbbbbbbb", "type": "llm"},
                {"number": 3, "title": "Cccccccccc", "type": "llm"},
                {"number": 4, "title": "Dddddddddd", "type": "llm"},
            ],
            "edges": [
                {"from": 1, "to": 2}, {"from": 2, "to": 3},
                {"from": 3, "to": 4}, {"from": 1, "to": 4},
            ],
        }, max_graph_width=12)
        _, legend = TestCompactMode()._split(lines)
        legend_text = "\n".join(legend)
        assert "4 — Dddddddddd (AI)  ◀ 1" in legend_text


class TestCrossingMinimization:
    def test_crossed_edges_get_uncrossed(self):
        # Two parents → two children with deliberately crossed edges
        # (1→4, 2→3). A single downward pass already fixes this shape, but
        # the ordering must place child 4 (under parent 1) left of child 3
        # (under parent 2) so the lines don't cross.
        out = _joined({
            "nodes": [
                {"number": 1, "title": "P one", "type": "llm"},
                {"number": 2, "title": "P two", "type": "llm"},
                {"number": 3, "title": "C three", "type": "llm"},
                {"number": 4, "title": "D four", "type": "llm"},
            ],
            "edges": [
                {"from": 1, "to": 4},
                {"from": 2, "to": 3},
            ],
        })
        child_row = next(
            ln for ln in out.splitlines() if "4 · D four" in ln
        )
        assert "3 · C three" in child_row
        assert child_row.index("4 · D four") < child_row.index("3 · C three")

    def test_deep_crossing_resolved_by_up_sweep(self):
        # A shape where the optimal middle-layer order depends on the
        # layer *below* it — only the upward sweep can see it. Renders
        # without hanging and keeps every node.
        out = _joined({
            "nodes": [
                {"number": 1, "title": "root", "type": "llm"},
                {"number": 2, "title": "mid a", "type": "llm"},
                {"number": 3, "title": "mid b", "type": "llm"},
                {"number": 4, "title": "leaf x", "type": "llm"},
                {"number": 5, "title": "leaf y", "type": "llm"},
            ],
            "edges": [
                {"from": 1, "to": 2}, {"from": 1, "to": 3},
                {"from": 2, "to": 5}, {"from": 3, "to": 4},
            ],
        })
        for label in ("root", "mid a", "mid b", "leaf x", "leaf y"):
            assert label in out


class TestStyled:
    """The colour-tinted twin renderer (render_dag_styled)."""

    STEPS = {
        "steps": [
            {"number": 1, "title": "Analyse query", "type": "llm",
             "dependencies": []},
            {"number": 2, "title": "Fetch data", "type": "tool",
             "dependencies": [1]},
            {"number": 3, "title": "Summarise", "type": "llm",
             "dependencies": [2]},
        ]
    }

    DIAMOND = {
        "nodes": [
            {"id": 1, "name": "root", "type": "llm"},
            {"id": 2, "name": "left", "type": "tool"},
            {"id": 3, "name": "right", "type": "mcp"},
            {"id": 4, "name": "merge", "type": "code"},
        ],
        "edges": [
            {"from": 1, "to": 2}, {"from": 1, "to": 3},
            {"from": 2, "to": 4}, {"from": 3, "to": 4},
        ],
    }

    def test_returns_text_lines(self):
        out = render_dag_styled(self.STEPS)
        assert out and all(isinstance(t, Text) for t in out)

    def test_empty_payload_returns_empty(self):
        assert render_dag_styled({}) == []
        assert render_dag_styled(None) == []

    def test_layout_matches_plain_renderer(self):
        # The styled twin must lay out byte-for-byte like the plain one —
        # only colour is added. Holds across linear, diamond, compact and
        # cycle paths.
        for payload, kw in [
            (self.STEPS, {}),
            (self.DIAMOND, {}),
            (TestCompactMode.WIDE, {"max_graph_width": 40}),
            ({"steps": [
                {"id": "a", "type": "llm", "deps": ["b"]},
                {"id": "b", "type": "tool", "deps": ["a"]},
            ]}, {}),
        ]:
            styled = render_dag_styled(payload, **kw)
            plain = render_dag_boxes(payload, **kw)
            assert [t.plain for t in styled] == plain

    def test_type_colours_applied_without_status(self):
        # No status map → each box tinted by its step type.
        styles = _styles(render_dag_styled(self.DIAMOND))
        assert "cyan" in styles      # llm  (root)
        assert "magenta" in styles   # tool (left)
        assert "blue" in styles      # mcp  (right)
        assert "green" in styles     # code (merge)

    def test_status_colours_override_type(self):
        out = render_dag_styled(
            self.STEPS, status_by_ref={"1": "done", "2": "running"},
        )
        styles = _styles(out)
        assert "green" in styles        # step 1 done
        assert "bold yellow" in styles  # step 2 running
        # step 3 absent from the map → pending → muted, not its llm cyan.
        assert "grey50" in styles
        assert "cyan" not in styles

    def test_failed_status_is_red(self):
        out = render_dag_styled(self.STEPS, status_by_ref={"2": "failed"})
        assert "bold red" in _styles(out)

    def test_cycle_fallback_is_styled(self):
        out = render_dag_styled({"steps": [
            {"id": "a", "type": "llm", "deps": ["b"]},
            {"id": "b", "type": "tool", "deps": ["a"]},
        ]})
        joined = "\n".join(t.plain for t in out)
        assert "cycle detected" in joined
        # The cycle marker rides a red style.
        assert "bold red" in _styles(out)

    def test_compact_legend_ref_is_tinted(self):
        # In compact mode the legend's leading ref number colour-matches
        # its box so the two views cross-reference by colour.
        out = render_dag_styled(TestCompactMode.WIDE, max_graph_width=40)
        assert _styles(out)  # legend + graph carry tints
        joined = "\n".join(t.plain for t in out)
        assert "steps:" in joined


class TestWrapping:
    """2-line label wrapping (max_lines) for the roomy modal pane."""

    LONG = {
        "steps": [
            {"number": 1, "type": "llm", "dependencies": [],
             "title": "Analyse the incoming customer support ticket"},
            {"number": 2, "type": "llm", "dependencies": [1], "title": "Reply"},
        ]
    }

    def test_default_truncates_with_ellipsis(self):
        assert "…" in _joined(self.LONG, max_width=20)

    def test_max_lines_wraps_instead_of_truncating(self):
        out = render_dag_boxes(self.LONG, max_width=20, max_lines=2)
        joined = "\n".join(out)
        # A continuation row carries later words of the long title.
        assert "customer" in joined
        # The wrapped layout is taller than the single-line one.
        assert len(out) > len(render_dag_boxes(self.LONG, max_width=20))

    def test_styled_wrap_matches_plain(self):
        styled = render_dag_styled(self.LONG, max_width=20, max_lines=2)
        plain = render_dag_boxes(self.LONG, max_width=20, max_lines=2)
        assert [t.plain for t in styled] == plain

    def test_short_labels_keep_single_line_height(self):
        short = {"steps": [{"number": 1, "title": "Hi", "type": "llm"}]}
        # Nothing needs wrapping → identical to max_lines=1.
        assert render_dag_boxes(short, max_lines=2) == render_dag_boxes(short)


class TestAsciiFallback:
    G = {
        "steps": [
            {"number": 1, "title": "A", "type": "llm", "dependencies": []},
            {"number": 2, "title": "B", "type": "tool", "dependencies": [1]},
            {"number": 3, "title": "C", "type": "mcp", "dependencies": [1]},
        ]
    }

    def test_no_unicode_glyphs_remain(self):
        out = _joined(self.G, ascii_only=True)
        for glyph in "┌┐└┘─│├┤┬┴┼▼◀…·—":
            assert glyph not in out
        assert "+" in out and "-" in out and "|" in out and "v" in out

    def test_width_preserved(self):
        uni = render_dag_boxes(self.G)
        asc = render_dag_boxes(self.G, ascii_only=True)
        assert [len(ln) for ln in uni] == [len(ln) for ln in asc]

    def test_styled_ascii_keeps_colour(self):
        lines = render_dag_styled(self.G, ascii_only=True)
        joined = "\n".join(t.plain for t in lines)
        assert "┌" not in joined and "+" in joined
        assert "cyan" in _styles(lines)  # type colour survives the swap


class TestDimming:
    G = {
        "steps": [
            {"number": 1, "title": "root", "type": "llm", "dependencies": []},
            {"number": 2, "title": "left", "type": "tool", "dependencies": [1]},
            {"number": 3, "title": "right", "type": "mcp", "dependencies": [1]},
            {"number": 4, "title": "merge", "type": "code", "dependencies": [2]},
        ]
    }

    def test_unrelated_node_is_dimmed(self):
        # Highlight 2 → lineage {1, 2, 4}; node 3 is unrelated → dimmed.
        styles = _styles(
            render_dag_styled(self.G, highlight_ref="2", dim_unrelated=True)
        )
        assert "grey37" in styles
        assert "bold underline magenta" in styles  # node 2 selected (tool)
        assert "blue" not in styles  # node 3's mcp tint replaced by dim

    def test_lineage_keeps_its_colour(self):
        styles = _styles(
            render_dag_styled(self.G, highlight_ref="2", dim_unrelated=True)
        )
        assert "cyan" in styles   # node 1 — llm ancestor
        assert "green" in styles  # node 4 — code descendant

    def test_no_dim_without_flag(self):
        styles = _styles(render_dag_styled(self.G, highlight_ref="2"))
        assert "grey37" not in styles
        assert "blue" in styles  # node 3 keeps its colour


class TestLeftToRight:
    """layout="lr" — transposed, left-to-right orientation."""

    LINEAR = {
        "steps": [
            {"number": 1, "title": "Plan", "type": "llm", "dependencies": []},
            {"number": 2, "title": "Fetch", "type": "tool",
             "dependencies": [1]},
            {"number": 3, "title": "Write", "type": "llm",
             "dependencies": [2]},
        ]
    }
    FORK = {
        "nodes": [
            {"number": 1, "title": "root", "type": "llm"},
            {"number": 2, "title": "left", "type": "tool"},
            {"number": 3, "title": "right", "type": "mcp"},
            {"number": 4, "title": "merge", "type": "llm"},
        ],
        "edges": [
            {"from": 1, "to": 2}, {"from": 1, "to": 3},
            {"from": 2, "to": 4}, {"from": 3, "to": 4},
        ],
    }

    def test_arrows_point_right_not_down(self):
        out = _joined(self.LINEAR, layout="lr")
        assert "▶" in out
        assert "▼" not in out

    def test_linear_chain_is_a_horizontal_band(self):
        out = render_dag_boxes(self.LINEAR, layout="lr")
        row = next(ln for ln in out if "1 · Plan" in ln)
        # All three boxes share the band → later steps live on the same row.
        assert "2 · Fetch" in row and "3 · Write" in row

    def test_fork_keeps_every_node(self):
        out = _joined(self.FORK, layout="lr")
        for label in ("root", "left", "right", "merge"):
            assert label in out

    def test_styled_lr_keeps_type_colour(self):
        lines = render_dag_styled(self.FORK, layout="lr")
        assert "magenta" in _styles(lines)   # the tool node
        assert any("▶" in t.plain for t in lines)

    def test_lr_layout_matches_plain(self):
        styled = render_dag_styled(self.FORK, layout="lr")
        plain = render_dag_boxes(self.FORK, layout="lr")
        assert [t.plain for t in styled] == plain

    def test_lr_skip_dep_keeps_annotation(self):
        out = _joined({
            "nodes": [
                {"number": 1, "title": "A", "type": "llm"},
                {"number": 2, "title": "B", "type": "llm"},
                {"number": 3, "title": "C", "type": "llm"},
            ],
            "edges": [
                {"from": 1, "to": 2}, {"from": 2, "to": 3}, {"from": 1, "to": 3},
            ],
        }, layout="lr")
        assert "◀ 1" in out


class TestBusLanes:
    """bus_lanes=True — draw skip edges as routed left-margin channels."""

    SKIP = {
        "nodes": [
            {"number": 1, "title": "A", "type": "llm"},
            {"number": 2, "title": "B", "type": "llm"},
            {"number": 3, "title": "C", "type": "llm"},
            {"number": 4, "title": "D", "type": "llm"},
        ],
        "edges": [
            {"from": 1, "to": 2}, {"from": 2, "to": 3}, {"from": 3, "to": 4},
            {"from": 1, "to": 4}, {"from": 2, "to": 4},
        ],
    }

    def test_default_annotates_skip_edges(self):
        assert "◀" in _joined(self.SKIP)

    def test_bus_lanes_draw_lines_not_annotations(self):
        out = _joined(self.SKIP, bus_lanes=True)
        assert "◀" not in out
        # Margin channels add box-drawing beyond the plain single spine.
        plain = _joined(self.SKIP)
        assert out.count("│") > plain.count("│")

    def test_bus_lanes_preserve_all_nodes(self):
        out = _joined(self.SKIP, bus_lanes=True)
        assert "1 · A" in out and "4 · D" in out

    def test_no_skip_edges_renders_identically(self):
        linear = {
            "steps": [
                {"number": 1, "title": "A", "type": "llm", "dependencies": []},
                {"number": 2, "title": "B", "type": "llm",
                 "dependencies": [1]},
            ]
        }
        assert render_dag_boxes(linear, bus_lanes=True) == render_dag_boxes(
            linear
        )

    def test_styled_bus_lanes_drop_annotation(self):
        lines = render_dag_styled(self.SKIP, bus_lanes=True)
        assert not any("◀" in t.plain for t in lines)

    def test_geometry_survives_margin_shift(self):
        geo: dict = {}
        render_dag_styled(self.SKIP, bus_lanes=True, geometry=geo)
        assert set(geo.values()) >= {"1", "2", "3", "4"}


class TestDisplayOpts:
    def test_reads_flags_off_config(self):
        from care.runtime.dag_view import dag_display_opts

        class _D:
            dag_ascii = True
            dag_bus_lanes = True

        class _C:
            defaults = _D()

        opts = dag_display_opts(_C())
        assert opts == {"ascii_only": True, "bus_lanes": True}

    def test_none_config_is_all_defaults(self):
        from care.runtime.dag_view import dag_display_opts

        assert dag_display_opts(None) == {
            "ascii_only": False, "bus_lanes": False,
        }


class TestDagDiff:
    """Version diff overlay (diff_chains / render_dag_diff)."""

    OLD = {
        "steps": [
            {"number": 1, "title": "Plan", "step_type": "llm",
             "dependencies": []},
            {"number": 2, "title": "Fetch", "step_type": "tool",
             "dependencies": [1]},
            {"number": 3, "title": "Summarise", "step_type": "llm",
             "dependencies": [2]},
        ]
    }
    # Insert "Validate" (renumbers), change Summarise's type.
    NEW = {
        "steps": [
            {"number": 1, "title": "Plan", "step_type": "llm",
             "dependencies": []},
            {"number": 2, "title": "Fetch", "step_type": "tool",
             "dependencies": [1]},
            {"number": 3, "title": "Validate", "step_type": "code",
             "dependencies": [2]},
            {"number": 4, "title": "Summarise", "step_type": "mcp",
             "dependencies": [3]},
        ]
    }

    def test_added_changed_unchanged_by_title(self):
        from care.runtime.dag_view import diff_chains

        status, removed = diff_chains(self.OLD, self.NEW)
        assert status["1"] == "unchanged"   # Plan
        assert status["2"] == "unchanged"   # Fetch (dep identity unchanged)
        assert status["3"] == "added"       # Validate
        assert status["4"] == "changed"     # Summarise: type llm→mcp
        assert removed == []

    def test_removed_step_detected(self):
        from care.runtime.dag_view import diff_chains

        new = {"steps": [
            {"number": 1, "title": "Plan", "step_type": "llm",
             "dependencies": []},
            {"number": 2, "title": "Summarise", "step_type": "llm",
             "dependencies": [1]},
        ]}
        status, removed = diff_chains(self.OLD, new)
        assert [label for _, label in removed] and any(
            "Fetch" in label for _, label in removed
        )

    def test_dependency_renumber_not_flagged_changed(self):
        from care.runtime.dag_view import diff_chains

        old = {"steps": [
            {"number": 1, "title": "A", "step_type": "llm", "dependencies": []},
            {"number": 2, "title": "B", "step_type": "llm",
             "dependencies": [1]},
        ]}
        # Prepend X → A,B renumber and B's dep number shifts, but identity
        # (B depends on A) is unchanged.
        new = {"steps": [
            {"number": 1, "title": "X", "step_type": "llm", "dependencies": []},
            {"number": 2, "title": "A", "step_type": "llm",
             "dependencies": []},
            {"number": 3, "title": "B", "step_type": "llm",
             "dependencies": [2]},
        ]}
        status, _ = diff_chains(old, new)
        assert status["1"] == "added"       # X
        assert status["2"] == "unchanged"   # A
        assert status["3"] == "unchanged"   # B — dep still points at A

    def test_render_diff_colours_added_and_changed(self):
        styles = _styles(render_dag_diff(self.OLD, self.NEW))
        assert "bold green" in styles    # added
        assert "bold yellow" in styles   # changed

    def test_render_diff_lists_removed(self):
        new = {"steps": [
            {"number": 1, "title": "Plan", "step_type": "llm",
             "dependencies": []},
        ]}
        out = render_dag_diff(self.OLD, new)
        joined = "\n".join(t.plain for t in out)
        assert "removed:" in joined
        assert "Fetch" in joined and "Summarise" in joined


class TestHeatmap:
    G = {
        "steps": [
            {"number": 1, "title": "A", "type": "llm", "dependencies": []},
            {"number": 2, "title": "B", "type": "tool", "dependencies": [1]},
            {"number": 3, "title": "C", "type": "llm", "dependencies": [2]},
        ]
    }

    def test_metric_maps_to_heat_scale(self):
        styles = _styles(
            render_dag_styled(self.G, metric_by_ref={"1": 0.0, "2": 5.0,
                                                     "3": 10.0})
        )
        assert "green" in styles      # low
        assert "yellow" in styles     # mid
        assert "bold red" in styles   # high

    def test_missing_metric_is_dimmed(self):
        styles = _styles(
            render_dag_styled(self.G, metric_by_ref={"1": 3.0})
        )
        assert "grey37" in styles     # steps 2,3 have no metric → dim

    def test_metric_overrides_type_colour(self):
        styles = _styles(
            render_dag_styled(self.G, metric_by_ref={"1": 0.0, "2": 5.0,
                                                     "3": 10.0})
        )
        assert "cyan" not in styles and "magenta" not in styles


class TestProfilingMetric:
    def test_projects_time_by_step_ref(self):
        from care.profiling import ProfilingSummary, StepProfile, profiling_metric

        summ = ProfilingSummary(steps=(
            StepProfile(1, "A", "llm", 0.5, 0, 0, 0, None),
            StepProfile(2, "B", "tool", 2.0, 0, 0, 0, None),
        ))
        assert profiling_metric(summ, "time") == {"1": 0.5, "2": 2.0}


class TestMermaid:
    G = {
        "steps": [
            {"number": 1, "title": "Plan", "type": "llm", "dependencies": []},
            {"number": 2, "title": "Fetch", "type": "tool",
             "dependencies": [1]},
            {"number": 3, "title": "Write", "type": "llm",
             "dependencies": [1, 2]},
        ]
    }

    def test_flowchart_header_and_nodes(self):
        from care.runtime.dag_view import render_dag_mermaid

        out = render_dag_mermaid(self.G)
        assert out.startswith("flowchart TD")
        assert 'n1["1 · Plan (AI)"]' in out
        assert 'n2["2 · Fetch (Tool)"]' in out

    def test_edges_rendered(self):
        from care.runtime.dag_view import render_dag_mermaid

        out = render_dag_mermaid(self.G)
        assert "n1 --> n2" in out
        assert "n1 --> n3" in out
        assert "n2 --> n3" in out

    def test_lr_direction(self):
        from care.runtime.dag_view import render_dag_mermaid

        assert render_dag_mermaid(self.G, layout="lr").startswith(
            "flowchart LR"
        )

    def test_quotes_escaped_in_label(self):
        from care.runtime.dag_view import render_dag_mermaid

        out = render_dag_mermaid(
            {"steps": [{"number": 1, "title": 'a"b"c', "type": "llm"}]}
        )
        # The only double-quotes left are the two wrapping the label.
        assert out.count('"') == 2

    def test_empty_payload(self):
        from care.runtime.dag_view import render_dag_mermaid

        assert render_dag_mermaid({}) == ""
        assert render_dag_mermaid(None) == ""
