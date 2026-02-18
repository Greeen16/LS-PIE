def paperExamplegen(N):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal


    t = np.linspace(0, 1, 800)

    sawtooths = []
    sinusoids = []

    plt.plot(t, signal.sawtooth(2 * np.pi * 3 * t))
    plt.plot(t, np.sin(2 * np.pi  * t))
    plt.show()
    
    noiseScale = 0.3
    halftot = int(np.round(N/2))
    #print(halftot)

    for i in range(halftot):
        scaler = np.random.random(1)
        noise = np.random.random(len(t))*noiseScale

        saw = signal.sawtooth(2 * np.pi * 3 * t)*scaler+noise
        sin = np.sin(2 * np.pi  * t)*scaler +noise

        sawtooths.append(saw)
        sinusoids.append(sin)

    data = sawtooths+sinusoids
    import numpy as np
    data = np.array(data).T
    
    return data,t

    # second example function
def paperExamplegen2(N):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    import scipy


    t = np.linspace(0, 1, 800)

    #sawtooths = []
    signals = []

    plt.plot(t, signal.sawtooth(2 * np.pi * 3 * t))
    plt.plot(t, np.sin(2 * np.pi  * t))
    plt.show()
    noiseScale = 0.3
    halftot = int(np.round(N/2))
    #print(halftot)

    out = []
    for i in range(halftot):
        scaler1 = np.random.random(1)
        scaler2 = np.random.random(1)

        noise = np.random.random(len(t))*noiseScale/2

        saw = scaler1*scipy.signal.sawtooth(2 * np.pi * 3 * t)+noise
        sin = scaler2*np.sin(2 * np.pi  * t)+noise

        signal = saw+sin
        out.append(signal)
    import numpy as np
    data = np.array(out).T
    return data,t
