from onpolicy.envs.agar.gamemodes.Mode import Mode

class FFA(Mode):
    def __init__(self):
        Mode.__init__(self)
        self.ID = 0
        self.name = "Free For All"
        self.specByLeaderboard = True


    def onPlayerSpawn(self, gameServer, player, pos=None):
        player.color = gameServer.getRandomColor() # Random color
        if pos is not None:
            gameServer.spawnPlayer(player, pos, False)
        else:
            gameServer.spawnPlayer(player, gameServer.randomPos2())