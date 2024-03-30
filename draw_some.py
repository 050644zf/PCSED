import matplotlib.pyplot as plt
import scipy.io as sio

# 读取 .mat 文件
data = sio.loadmat(r'./nets/hybnet/1.科学级-自动生成9/TargetCurves.mat')


your_data = data['TargetCurves']

# 绘制九条曲线
for i in range(9):
    plt.plot(your_data[i, :], label=f'Curve {i + 1}')

# 设置图例和标题
plt.legend()
plt.title('Nine Curves from TargetCurves.mat')

# 显示图形
plt.show()


