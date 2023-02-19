class jetNode():
    def __init__(self, vec4, left = None, right = None, decay_rate = 0, delta = 0, logLH = 0, dijList = []):
        self.vec4 = vec4
        self.left = left
        self.right = right
        self.decay_rate = decay_rate
        self.delta = delta # invariant mass squared

        self.logLH = logLH
        self.dijList = dijList