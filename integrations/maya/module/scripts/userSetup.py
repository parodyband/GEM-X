# GEM-X Live: load the plug-in when Maya starts (adds the GEM-X menu).
def _gemx_live_autoload():
    import maya.cmds as cmds

    if not cmds.about(batch=True) and not cmds.pluginInfo("gemxLive", q=True, loaded=True):
        try:
            cmds.loadPlugin("gemxLive", quiet=True)
        except RuntimeError as e:
            print(f"GEM-X Live: could not load the plug-in: {e}")


import maya.utils

maya.utils.executeDeferred(_gemx_live_autoload)
