"""Read-only NetworkX rendering for the live mesh painter."""

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path


class LiveKnowledgeGraph:
    """Render Priority Map's spatial graph directly from graph.db."""

    def __init__(self, graph_db):
        self.path = Path(graph_db).resolve()
        uri = self.path.as_uri() + "?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        self.figure = None
        self.axis = None
        self.window_process = None
        self._data_version = None

    def _read(self):
        rows = self.connection.execute(
            """
            SELECT id, label, score, color_b, color_g, color_r,
                   geo_pos_x, geo_pos_y
            FROM nodes
            """
        ).fetchall()
        nodes = {
            str(node_id): {
                "label": str(label),
                "score": float(score),
                "rgb": (int(red), int(green), int(blue)),
                "pos": (float(x), float(y)),
            }
            for node_id, label, score, blue, green, red, x, y in rows
        }
        edge_rows = self.connection.execute(
            "SELECT source_id, target_id, weight FROM edges"
        ).fetchall()
        edges = []
        for source_id, target_id, weight in edge_rows:
            source_id = str(source_id)
            target_id = str(target_id)
            if source_id not in nodes or target_id not in nodes:
                continue
            edges.append((source_id, target_id, float(weight)))
        return nodes, edges

    def _draw(self, axis):
        """Draw GraphBuilder's spatial-graph conventions onto an axes."""
        import networkx as nx

        nodes, spatial_edges = self._read()
        if not nodes:
            axis.clear()
            axis.set_axis_off()
            return False

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_weighted_edges_from(spatial_edges)
        if graph.number_of_edges():
            graph = nx.minimum_spanning_tree(graph, weight="weight")
        positions = {node_id: data["pos"] for node_id, data in nodes.items()}
        edge_labels = {
            (source_id, target_id): str(int(round(data.get("weight", 0))))
            for source_id, target_id, data in graph.edges(data=True)
        }

        axis.clear()
        node_sizes = [100 + nodes[node_id]["score"] / 100.0 * 1000 for node_id in graph]
        node_colors = [
            tuple(channel / 255.0 for channel in nodes[node_id]["rgb"])
            for node_id in graph
        ]
        labels = {node_id: nodes[node_id]["label"] for node_id in graph}
        nx.draw(
            graph,
            positions,
            ax=axis,
            labels=labels,
            with_labels=True,
            node_size=node_sizes,
            node_color=node_colors,
            font_size=14,
        )
        if edge_labels:
            nx.draw_networkx_edge_labels(
                graph,
                positions,
                edge_labels=edge_labels,
                ax=axis,
                font_size=10,
                rotate=False,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 1,
                },
            )
        axis.set_axis_off()
        return True

    def render(self):
        """Return an RGB snapshot for tests and noninteractive consumers."""
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8, 5))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        if not self._draw(axis):
            return None
        figure.canvas.draw()
        image = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
        return image

    def open_window(self, position, size):
        """Launch a responsive native Matplotlib window in a companion process."""
        x, y = position
        width, height = size
        self.window_process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(self.path),
                "--x",
                str(x),
                "--y",
                str(y),
                "--width",
                str(width),
                "--height",
                str(height),
            ]
        )

    def _open_native_window(self, position, size):
        """Run the native graph UI and poll SQLite for external commits."""
        import matplotlib.pyplot as plt

        plt.ioff()
        self.figure, self.axis = plt.subplots(figsize=(8, 5))
        manager = self.figure.canvas.manager
        if hasattr(manager, "set_window_title"):
            manager.set_window_title("Priority Map - Spatial Knowledge Graph")
        window = getattr(manager, "window", None)
        x, y = position
        width, height = size
        if hasattr(window, "wm_geometry"):
            window.wm_geometry(f"{width}x{height}+{x}+{y}")
        elif hasattr(window, "setGeometry"):
            window.setGeometry(x, y, width, height)
        self._refresh_native_window()
        self._data_version = self.connection.execute("PRAGMA data_version").fetchone()[0]
        manager.show()
        while self.figure is not None and plt.fignum_exists(self.figure.number):
            self._poll_database()
            plt.pause(0.25)

    def _poll_database(self):
        if self.figure is None:
            return False
        version = self.connection.execute("PRAGMA data_version").fetchone()[0]
        if version != self._data_version:
            self._data_version = version
            self._refresh_native_window()
        return True

    def refresh_window(self):
        """The companion process observes committed changes through SQLite."""
        return

    def _refresh_native_window(self):
        if self.figure is None or self.axis is None:
            return
        if not self._draw(self.axis):
            return
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def close(self):
        if self.window_process is not None:
            if self.window_process.poll() is None:
                self.window_process.terminate()
                try:
                    self.window_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.window_process.kill()
            self.window_process = None
        if self.figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self.figure)
            self.figure = None
            self.axis = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def main():
    parser = argparse.ArgumentParser(description="Show a live Priority Map spatial graph")
    parser.add_argument("graph_db")
    parser.add_argument("--x", type=int, default=900)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()
    graph = LiveKnowledgeGraph(args.graph_db)
    try:
        graph._open_native_window(
            (args.x, args.y),
            (args.width, args.height),
        )
    finally:
        graph.close()


if __name__ == "__main__":
    main()
