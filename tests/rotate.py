import subprocess, tempfile, textwrap
from pathlib import Path
from typing import Iterable, Tuple, Literal


Rotation = Literal["ROTATE_NONE", "ROTATE_90", "ROTATE_180", "ROTATE_270"]


def create_qupath_project_rotated(
    project_dir: Path,
    images: Iterable[Tuple[Path, Rotation]]  # (image_path, rotation)
) -> None:
    """
    Create a QuPath project & add images, rotating each one as specified.

    rotation must be one of:
        ROTATE_NONE | ROTATE_90 | ROTATE_180 | ROTATE_270
    """
    project_dir = project_dir.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    qupath_exe = r"C:\Users\MIT\AppData\Local\QuPath-0.5.1\QuPath-0.5.1.exe"


    # --- Groovy snippet ---------------------------------------------------
    groovy = textwrap.dedent(r"""
        import qupath.lib.projects.ProjectIO
        import qupath.lib.images.servers.ImageServerProvider
        import qupath.lib.images.servers.TransformedServerBuilder
        import qupath.lib.images.servers.RotatedImageServer.Rotation

        def projDir = new File(args[0])
        def project = ProjectIO.createProject(projDir, null)

        // remaining args arrive as: imgPath1, rotStr1, imgPath2, rotStr2, ...
        for (int i = 1; i < args.length; i += 2) {
            def f        = new File(args[i])
            def rotEnum  = Rotation.valueOf(args[i + 1])   // e.g. ROTATE_90
            def baseSrv  = ImageServerProvider.buildServer(f.toURI(), false)
            def rotSrv   = new TransformedServerBuilder(baseSrv).rotate(rotEnum).build()  // :contentReference[oaicite:0]{index=0}

            project.addImage(rotSrv)
        }
        ProjectIO.writeProject(project)   // flush .qpproj & .qpdata
    """)

    with tempfile.NamedTemporaryFile(suffix=".groovy", delete=False) as f:
        f.write(groovy.encode())
        script_path = f.name

    # flatten CLI --args: [project, img1, rot1, img2, rot2, …]
    cli_args = [str(project_dir)]
    for img, rot in images:
        cli_args.extend([str(img.resolve()), rot])

    cmd = [
        qupath_exe, "script",
        "--headless",
        "--save",                 # write the project to disk
        "-c", groovy,             # pass Groovy inline
        *sum([["--args", a] for a in cli_args], []),
    ]

    subprocess.run(cmd, shell=True)

# ---------------- Example ----------------
create_qupath_project_rotated(
    Path(r"C:\Users\MIT\Desktop\qpp"),
    [
        (Path(r"C:\Users\MIT\Documents\Dirigo\experiment.ome.tif"), "ROTATE_90"),
    ]
)
