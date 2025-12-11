import os
import subprocess

from src.configs import load_paths_config


def run_gpmetis() -> str:
    """
    Run gpmetis on the exported METIS graph.

    Uses:
      paths.metis.work_dir
      paths.metis.graph_filename
      paths.metis.num_parts
      paths.metis.gpmetis_binary

    Output (by gpmetis):
      contracted.graph.part.<K> in the metis work_dir

    Returns:
      Path to the partition file.
    """
    paths = load_paths_config()

    metis_dir = paths.metis_work_dir
    graph_filename = paths.metis_graph_filename
    num_parts = paths.metis_num_parts
    gpmetis_bin = paths.metis_gpmetis_binary

    graph_path = os.path.join(metis_dir, graph_filename)
    part_file = f"{graph_filename}.part.{num_parts}"
    part_path = os.path.join(metis_dir, part_file)

    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"METIS graph file not found at {graph_path}. "
            f"Run export_to_metis first."
        )

    if not os.path.exists(gpmetis_bin):
        raise FileNotFoundError(
            f"gpmetis binary not found at {gpmetis_bin}. "
            f"Install METIS (e.g., 'brew install metis') and update paths.yaml."
        )

    cmd = [gpmetis_bin, graph_path, str(num_parts)]
    print(f"Running gpmetis: {' '.join(cmd)} (cwd={metis_dir})")

    result = subprocess.run(
        cmd,
        cwd=metis_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("gpmetis stderr:")
        print(result.stderr)
        raise RuntimeError(f"gpmetis failed with return code {result.returncode}")

    if not os.path.exists(part_path):
        raise FileNotFoundError(
            f"Expected partition file not found: {part_path}"
        )

    print(f"gpmetis completed. Partition file: {part_path}")
    return part_path


if __name__ == "__main__":
    run_gpmetis()
