"""GEM-X Live: markerless live motion capture for Maya.

Load from the Plug-in Manager (gemxLive.py). Adds a GEM-X menu and the
commands:

    gemxLive [-close]
        Open (or close) the GEM-X Live window.

    gemxBakeTake -take <path> -root <joint> [-start <frame>]
        Retarget a recorded take onto a character and key it. Undoable.
        Uses the mapping, mask and root options stored on the character.
"""

import maya.api.OpenMaya as om
import maya.cmds as cmds

PLUGIN_VENDOR = "parodyband"
PLUGIN_VERSION = "1.0.0"
MENU_NAME = "gemxLiveMenu"
HELP_URL = "https://github.com/parodyband/GEM-X/tree/main/integrations/maya#readme"


def maya_useNewAPI():
    pass


class GemxLiveCmd(om.MPxCommand):
    NAME = "gemxLive"

    @staticmethod
    def creator():
        return GemxLiveCmd()

    @staticmethod
    def syntax():
        s = om.MSyntax()
        s.addFlag("-c", "-close")
        return s

    def doIt(self, args):
        db = om.MArgDatabase(self.syntax(), args)
        from gemx_live import ui

        if db.isFlagSet("-c"):
            ui.close()
        else:
            ui.show()


class GemxBakeTakeCmd(om.MPxCommand):
    NAME = "gemxBakeTake"

    def __init__(self):
        super().__init__()
        self._writer = None

    @staticmethod
    def creator():
        return GemxBakeTakeCmd()

    @staticmethod
    def syntax():
        s = om.MSyntax()
        s.addFlag("-t", "-take", om.MSyntax.kString)
        s.addFlag("-r", "-root", om.MSyntax.kString)
        s.addFlag("-s", "-start", om.MSyntax.kDouble)
        return s

    def isUndoable(self):
        return True

    def doIt(self, args):
        db = om.MArgDatabase(self.syntax(), args)
        if not (db.isFlagSet("-t") and db.isFlagSet("-r")):
            raise RuntimeError("gemxBakeTake needs -take <path> and -root <joint>")
        take_path = db.flagArgumentString("-t", 0)
        root = db.flagArgumentString("-r", 0)
        start = db.flagArgumentDouble("-s", 0) if db.isFlagSet("-s") else cmds.currentTime(q=True)

        from gemx_live import bake
        from gemx_live.takes import Take
        from gemx_live.target import Character

        take = Take.load(take_path)
        character = Character(root)
        if not character.mapping:
            raise RuntimeError(f"{root} has no GEM-X mapping; set it up in the GEM-X Live window first")
        fps = om.MTime(1.0, om.MTime.kSeconds).asUnits(om.MTime.uiUnit())
        data = bake.compute(take, character, start, fps)
        if not data.channels:
            raise RuntimeError("nothing to key: every mapped joint is masked, locked or constrained")
        for name in data.skipped:
            om.MGlobal.displayWarning(f"gemxBakeTake: skipped {name} (locked or driven by a non-animation connection)")
        self._writer = bake.KeyWriter()
        keys = self._writer.write(data)
        self.setResult(keys)
        om.MGlobal.displayInfo(
            f"gemxBakeTake: keyed {take.name} on {character.names[0]} "
            f"(frames {data.frames[0]:g}-{data.frames[-1]:g}, {keys} keys)"
        )

    def undoIt(self):
        if self._writer:
            self._writer.undo()

    def redoIt(self):
        if self._writer:
            self._writer.redo()


COMMANDS = (GemxLiveCmd, GemxBakeTakeCmd)


def _build_menu():
    if om.MGlobal.mayaState() != om.MGlobal.kInteractive:
        return
    if cmds.menu(MENU_NAME, exists=True):
        cmds.deleteUI(MENU_NAME)
    cmds.menu(MENU_NAME, label="GEM-X", parent="MayaWindow", tearOff=True)
    cmds.menuItem(label="Live Capture...", annotation="Open the GEM-X Live window",
                  command=lambda *_: cmds.gemxLive())
    cmds.menuItem(divider=True)
    cmds.menuItem(label="Close Live Capture", command=lambda *_: cmds.gemxLive(close=True))
    cmds.menuItem(divider=True)
    cmds.menuItem(label="Help...", command=lambda *_: cmds.showHelp(HELP_URL, absolute=True))


def initializePlugin(plugin):
    fn = om.MFnPlugin(plugin, PLUGIN_VENDOR, PLUGIN_VERSION, "Any")
    for cmd in COMMANDS:
        fn.registerCommand(cmd.NAME, cmd.creator, cmd.syntax)
    _build_menu()


def uninitializePlugin(plugin):
    fn = om.MFnPlugin(plugin)
    try:
        from gemx_live import ui

        ui.close(shutdown=True)
    except Exception as e:  # never block unloading
        om.MGlobal.displayWarning(f"GEM-X Live: {e}")
    if om.MGlobal.mayaState() == om.MGlobal.kInteractive and cmds.menu(MENU_NAME, exists=True):
        cmds.deleteUI(MENU_NAME)
    for cmd in reversed(COMMANDS):
        fn.deregisterCommand(cmd.NAME)
    # Forget the package so reloading the plug-in picks up updated code.
    import sys

    for name in [m for m in sys.modules if m == "gemx_live" or m.startswith("gemx_live.")]:
        del sys.modules[name]
