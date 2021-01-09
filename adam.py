import numpy as np

def gradient_descent(df, x0, step_size = 0.001, grad_threshold = 0.001, max_iteration = 1000):
    """標準的な勾配法
    パラメータ
    df: 目的関数の導関数
    x0: 初期値
    step_size: 入力値の変化量をスケールする係数、学習率
    grad_threshold: 勾配を0とみなす閾値
    max_iteration: 反復回数の上限

    戻り値
        目的関数の値が最小となる入力値と実行された反復の回数のタプル
    """
    x = np.array(x0)
    t = 0
    while t < max_iteration:
        grad = np.array(df(x))
        if (np.abs(grad) < grad_threshold).all():
            break
        t = t + 1
        x = x - step_size * grad
    return x, t

def adam(df, x0, step_size = 0.001, decay_rate1 = 0.9, decay_rate2 = 0.999, epsilon = 0.00000001, grad_threshold = 0.001, max_iteration = 1000):
    """Adamによる勾配法
    パラメータ
    df: 目的関数の導関数
    x0: 初期値
    step_size, decay_rate1, decay_rate2, epsilon: 論文参照
    grad_threshold: 勾配を0とみなす閾値
    max_iteration: 反復回数の上限

    戻り値
        目的関数の値が最小となる入力値と実行された反復の回数のタプル
    """
    a, b1, b2, e = step_size, decay_rate1, decay_rate2, epsilon
    b1_t = 1
    b2_t = 1
    x = np.array(x0)
    m = np.zeros(x.shape)
    v = np.zeros(x.shape)
    t = 0
    while t < max_iteration:
        g = np.array(df(x))
        if (np.abs(g) < grad_threshold).all():
            break
        t = t + 1
        tmp = (1 - b1) * g
        m = b1 * m + tmp
        v = b2 * v + tmp * g
        b1_t = b1_t * b1
        b2_t = b2_t * b2
        m_hat = m / (1 - b1_t)
        v_hat = v / (1 - b2_t)
        x = x - a * m_hat / (np.sqrt(v_hat) + e)
    return x, t

#####################################################
# 以下、移動・回転する前とした後の点列の間の二乗距離を最小化する
# 移動・回転量を勾配法とAdamのそれぞれで求める例

def derivative_f_wrt_a(x1, y1, x0, y0, u, a):
    """点(x1, y1)と、点(x0, y0)を(u, a)だけ移動・回転させた点との二乗距離のaについての勾配
    x1: 現在時刻の観測値のx座標
    y1: 現在時刻の観測値のY座標
    x0: 前時刻の観測値のX座標
    y0: 前時刻の観測値のY座標
    u: 前時刻からのX軸方向の変位
    a: 前時刻からのZ軸中心の回転量

    二乗距離は(x1-x0*cos(a)-y0*sin(a)-u)^2+(y1-x0*sin(a)-y0*cos(a)-v)^2とし、以下のサイトで微分した。
    https://ja.wolframalpha.com/input/?i=%28s-x*cos%28a%29-y*sin%28a%29-u%29%5E2%2B%28t-x*sin%28a%29-y*cos%28a%29-v%29%5E2+%E3%82%92a%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6%E5%BE%AE%E5%88%86%E3%81%99%E3%82%8B
    """
    sin_a = np.sin(a)
    cos_a = np.cos(a)
    cos_2a = np.cos(2 * a)
    return np.sum(2 * (sin_a * (x1 * x0 + y1 * y0 - u * x0) + cos_a * (-x1 * y0 - y1 * x0 + u * y0) + 2 * x0 * y0 * cos_2a))

def derivative_f_wrt_u(x1, y1, x0, y0, u, a):
    """点(x1, y1)と、点(x0, y0)を(u, a)だけ移動・回転させた点との二乗距離のuについての勾配
    x1: 現在時刻の観測値のx座標
    y1: 現在時刻の観測値のY座標
    x0: 前時刻の観測値のX座標
    y0: 前時刻の観測値のY座標
    u: 前時刻からのX軸方向の変位
    a: 前時刻からのZ軸中心の回転量

    二乗距離は(x1-x0*cos(a)-y0*sin(a)-u)^2+(y1-x0*sin(a)-y0*cos(a)-v)^2とし、以下のサイトで微分した。
    https://ja.wolframalpha.com/input/?i=%28s-x*cos%28a%29-y*sin%28a%29-u%29%5E2%2B%28t-x*sin%28a%29-y*cos%28a%29-v%29%5E2+%E3%82%92u%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6%E5%BE%AE%E5%88%86%E3%81%99%E3%82%8B
    """
    sin_a = np.sin(a)
    cos_a = np.cos(a)
    return np.sum(2 * (x0 * cos_a + y0 * sin_a - x1 + u))

def derivative_f(x1, y1, x0, y0, u, a):
    """点(x1, y1)と、点(x0, y0)を(u, a)だけ移動・回転させた点との二乗距離のuについてのuとaについての勾配
    x1: 現在時刻の観測値のx座標
    y1: 現在時刻の観測値のY座標
    x0: 前時刻の観測値のX座標
    y0: 前時刻の観測値のY座標
    u: 前時刻からのX軸方向の変位
    a: 前時刻からのZ軸中心の回転量
    """
    return [
        derivative_f_wrt_u(x1, y1, x0, y0, u, a),
        derivative_f_wrt_a(x1, y1, x0, y0, u, a)]

def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return 180 * rad / np.pi

# 真値
gt_u = 10
gt_a = deg2rad(5)

# N個の(x0, y0, 1)のデータセットpoints0
N=100
points0=np.concatenate((np.random.random((2,N)), np.ones((1,N))))

# (x1, y1, 1)のデータセットpoints1
sin_gt_a = np.sin(gt_a)
cos_gt_a = np.cos(gt_a)
M = np.array([[cos_gt_a,sin_gt_a,gt_u],[sin_gt_a,cos_gt_a,0],[0,0,1]])
points1=M.dot(points0)

# 標準的な勾配法
max_iteration = 1000000
x, t = gradient_descent(
    lambda x: derivative_f(points1[0], points1[1], points0[0], points0[1], x[0], x[1]),
    [9, 0], max_iteration = max_iteration)
x[1] = rad2deg(x[1])
print(x, t)

# Adam
x, t = adam(
    lambda x: derivative_f(points1[0], points1[1], points0[0], points0[1], x[0], x[1]),
    [9, 0], max_iteration = max_iteration)
x[1] = rad2deg(x[1])
print(x, t)
