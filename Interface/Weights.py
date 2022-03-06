
def weight_(df):

    num_ports = 5000
    all_weights = np.zeros((num_ports,len(df.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)
    df = df.dropna(axis=0)
    log_ret = np.log(df / df.shift(1))
    for x in range(num_ports):
        #weights
        weights = np.array(np.random.random(len(df.columns)))
        weights = weights / np.sum(weights)
        # Save weights
        all_weights[x, :] = weights
        # Expected return
        ret_arr[x] = np.sum((log_ret.mean() * weights * 256))
        # Expected volatility
        vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 256, weights)))
        # Sharpe Ratio
        sharpe_arr[x] = ret_arr[x] / vol_arr[x]

    #Best Portföy
    p = sharpe_arr.argmax()
    # Best Weight
    w = all_weights[p, :]

    return w
