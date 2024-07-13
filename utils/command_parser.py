# 使用 argparse 模块解析命令行参数的Python脚本，用于设置训练过程中的各种参数
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Visual Navigation')

    # 训练的阶段，可能的值有 train, eval, debug
    parser.add_argument(
        '--phase',
        type=str,
        default='train',
        help='train, eval or debug three choices'
    )
    # 一个episode的最大长度
    parser.add_argument(
        '--max-episode-length',
        type=int,
        default=30,
        metavar='M',
        help='maximum length of an episode (default: 100)',
    )
    # 是否使用共享统计数据的优化器
    parser.add_argument(
        '--shared-optimizer',
        default=True,
        metavar='SO',
        help='use an optimizer with shared statistics.',
    )

    # 选择优化器类型，可以是 SharedAdam 或 SharedRMSprop
    parser.add_argument(
        '--optimizer',
        default='SharedAdam',
        metavar='OPT',
        help='shared optimizer choice of SharedAdam or SharedRMSprop',
    )
    # Adam优化器的amsgrad参数
    parser.add_argument(
        '--amsgrad',
        default=True,
        metavar='AM',
        help='Adam optimizer amsgrad parameter'
    )
    # 用于离散化AI2-THOR地图的网格大小
    parser.add_argument(
        '--grid_size',
        type=float,
        default=0.25,
        metavar='GS',
        help='The grid size used to discretize AI2-THOR maps.',
    )
    # 是否启用Docker
    parser.add_argument(
        '--docker_enabled',
        action='store_true',
        help='Whether or not to use docker.'
    )
    # X显示目标
    parser.add_argument(
        '--x_display',
        type=str,
        default=None,
        help='The X display to target, if any.'
    )
    # 测试运行之间的等待时间
    parser.add_argument(
        '--test_timeout',
        type=int,
        default=10,
        help='The length of time to wait in between test runs.',
    )
    # 是否输出详细信息
    parser.add_argument(
        '--verbose',
        type=bool,
        default=False,
        help='If true, output will contain more information.',
    )
    # 打印信息的频率
    parser.add_argument(
        '--train_thin',
        type=int,
        default=1000,
        help='How often to print'
    )
    # 内存块的数量
    parser.add_argument(
        '--num-memory-block',
        type=int,
        default=2,
        help='the number of memory blocks'
    )
    # 本地AI2-THOR可执行文件的路径
    parser.add_argument(
        '--local_executable_path',
        type=str,
        default=None,
        help='a path to the local thor build.',
    )
    # 是否使用hindsight replay
    parser.add_argument(
        '--hindsight_replay',
        type=bool,
        default=False,
        help='whether or not to use hindsight replay.',
    )
    # 是否启用测试代理
    parser.add_argument(
        '--enable_test_agent',
        action='store_true',
        help='Whether or not to have a test agent.',
    )
    # 训练场景
    parser.add_argument(
        '--train_scenes',
        type=str,
        default='[1-20]',
        help='scenes for training.'
    )
    # 验证场景
    parser.add_argument(
        '--val_scenes',
        type=str,
        default='[21-30]',
        help='old validation scenes before formal split.',
    )
    # 所有可能的目标对象
    parser.add_argument(
        '--possible_targets',
        type=str,
        default='FULL_OBJECT_CLASS_LIST',
        help='all possible objects.',
    )
    # 实验中使用的特定对象
    # if none use all dest objects
    parser.add_argument(
        '--train_targets',
        type=str,
        default=None,
        help='specific objects for this experiment from the object list.',
    )
    # 可能动作的空间
    parser.add_argument(
        '--action_space',
        type=int,
        default=6,
        help='space of possible actions.'
    )
    # 注意力大小
    parser.add_argument(
        '--attention-sz',
        type=int,
        default=512,
    )
    # 是否计算spl
    parser.add_argument(
        '--compute_spl',
        action='store_true',
        help='compute the spl.'
    )
    # 运行测试时是否包含
    parser.add_argument(
        '--include-test',
        action='store_true',
        help='run test during eval'
    )
    # 是否严格完成
    parser.add_argument(
        '--disable-strict_done',
        dest='strict_done',
        action='store_false'
    )
   
    parser.set_defaults(strict_done=True)
    # 写入结果的文件
    parser.add_argument(
        '--results-json',
        type=str,
        default='metrics.json',
        help='Write the results.'
    )
    # 可视化文件的名称
    parser.add_argument(
        '--visualize-file-name',
        type=str,
        default='visual_temp.json'
    )
    # 代理类型，选择 NavigationAgent 或 RandomAgent
    parser.add_argument(
        '--agent_type',
        type=str,
        default='NavigationAgent',
        help='Which type of agent. Choices are NavigationAgent or RandomAgent.',
    )
    # episode类型
    parser.add_argument(
        '--episode_type',
        type=str,
        default='BasicEpisode',
        help='Which type of agent. Choices are NavigationAgent or RandomAgent.',
    )
    # 使用的视野
    parser.add_argument(
        '--fov',
        type=float,
        default=100.0,
        help='The field of view to use.'
    )
    # 场景类型
    parser.add_argument(
        '--scene-types',
        nargs='+',
        default=['kitchen', 'living_room', 'bedroom', 'bathroom'],
    )
    # MAML的梯度步骤限制
    parser.add_argument(
        '--gradient_limit',
        type=int,
        default=4,
        help='How many gradient steps allowed for MAML.',
    )
    # 测试或验证
    parser.add_argument(
        '--test_or_val',
        default='val',
        help='test or val'
    )
    # 从指定的episode开始测试
    parser.add_argument(
        '--test-start-from',
        type=int,
        default=0,
        help='start from given episode'
    )
    # 多头注意力的数量
    parser.add_argument(
        '--multi-heads',
        type=int,
        default=1
    )
    # 保留原始观察
    parser.add_argument(
        '--keep-ori-obs',
        action='store_true',
    )
    # 是否记录注意力
    parser.add_argument(
        '--record-attention',
        action='store_true',
    )
    # 是否测试速度
    parser.add_argument(
        '--test-speed',
        action='store_true'
    )

    # ==================================================
    # arguments with normal settings   一般设置
    # 随机种子
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='random seed (default: 1)'
    )
    # 使用的模型
    parser.add_argument(
        '--model',
        type=str,
        default='BaseModel',
        help='Model to use.'
    )
    # 是否运行测试代码
    parser.add_argument(
        '--eval',
        action='store_true',
        help='run the test code'
    )
    # 是否记录动作地图
    parser.add_argument(
        '--record-act-map',
        action='store_true',
    )
    # 使用的GPU，-1表示仅使用CPU
    parser.add_argument(
        '--gpu-ids',
        type=int,
        default=-1,
        nargs='+',
        help='GPUs to use [-1 CPU only] (default: -1)',
    )
    # 使用的训练进程数量
    parser.add_argument(
        '--workers',
        type=int,
        default=12,
        metavar='W',
        help='how many training processes to use (default: 4)',
    )
    # 日志记录的信息
    parser.add_argument(
        '--title',
        type=str,
        default='a3c',
        help='Info for logging.'
    )
    # 工作目录，包括TensorBoard日志目录、日志文件、训练模型等
    parser.add_argument(
        '--work-dir',
        type=str,
        default='./debugs/',
        help='Work directory, including: tensorboard log dir, log txt, trained models',
    )
    # 保存训练模型的文件夹
    parser.add_argument(
        '--save-model-dir',
        default='debugs',
        help='folder to save trained navigation',
    )
    # 最大的episode数量
    parser.add_argument(
        '--max-ep',
        type=float,
        default=6000000,
        help='maximum # of episodes'
    )
    # 每训练多少个episode保存一次模型
    parser.add_argument(
        '--ep-save-freq',
        type=int,
        default=1e5,
        help='save model after this # of training episodes (default: 1e+4)',
    )
    # 是否在训练后运行测试
    parser.add_argument(
        '--test-after-train',
        action='store_true',
        help='run test after training'
    )
    # 备注信息
    parser.add_argument(
        '--remarks',
        type=str,
        default=None,
    )
    # 是否禁用日志记录
    parser.add_argument(
        '--no-logger',
        action='store_true',
    )

    # arguments related with continue training based on existed trained models
    # 加载已保存模型的路径
    parser.add_argument(
        '--load-model',
        type=str,
        default=None,
        help='Path to load a saved model.'
    )
    # 基于给定模型继续训练
    parser.add_argument(
        '--continue-training',
        type=str,
        default=None,
        help='continue training based on given model'
    )
    # 基于给定模型进行微调
    parser.add_argument(
        '--fine-tuning',
        type=str,
        default=None,
        help='fine tune based on given model'
    )
    # 预训练转换器的路径
    parser.add_argument(
        '--pretrained-trans',
        type=str,
        default=None,
    )
    # 是否进行预热
    parser.add_argument(
        '--warm-up',
        action='store_true',
    )

    # arguments related with pretraining Visual Transformer
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
    )

    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
    )

    parser.add_argument(
        '--epoch-save',
        type=int,
        default=5,
    )

    # ==================================================
    # arguments related with data
    # 数据集存储路径
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/tmp/data/AI2Thor_Dataset/',
        help='where dataset is stored.',
    )
    # 类别数量
    parser.add_argument(
        '--num-category',
        type=int,
        default=22,
        help='the number of categories'
    )
    # 图文件名
    parser.add_argument(
        '--graph-file',
        type=str,
        default='graph.json',
        help='the name of the graph file'
    )
    # 网格文件名
    parser.add_argument(
        '--grid-file',
        type=str,
        default='grid.json',
        help='the name of the grid file'
    )
    # 可见对象地图文件名
    parser.add_argument(
        '--visible-map-file-name',
        type=str,
        default='visible_object_map_1.5.json',
        help='the name of the visible object map file'
    )
    # 检测算法选择 (fasterrcnn, detr, fasterrcnn_bottom)
    parser.add_argument(
        '--detection-alg',
        type=str,
        default='detr',
        choices=['fasterrcnn', 'detr', 'fasterrcnn_bottom']
    )
    # 存储检测特征的文件
    parser.add_argument(
        '--detection-feature-file-name',
        type=str,
        default=None,
        help='Which file store the detection feature?'
    )
    # 存储图像的文件名，可以是实际图像或Resnet特征
    parser.add_argument(
        '--images-file-name',
        type=str,
        default='resnet18_featuremap.hdf5',
        help='Where the controller looks for images. Can be switched out to real images or Resnet features.',
    )
    # 存储最佳动作的文件名
    parser.add_argument(
        '--optimal-action-file-name',
        type=str,
        default='optimal_action.json'
    )

    # ==================================================
    # arguments related with models
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        metavar='LR',
        help='learning rate (default: 0.0001)',
    )
    # 预训练学习率
    parser.add_argument(
        '--pretrained-lr',
        type=float,
        default=0.00001,
    )
    # 是否不使用位置增强
    parser.add_argument(
        '--wo-location-enhancement',
        action='store_true',
    )
    # 权重衰减
    parser.add_argument(
        '--weight-decay',
        default = 1e-4,
        type = float,
    )
    # 内部学习率
    parser.add_argument(
        '--inner-lr',
        type=float,
        default=0.0001,
        metavar='ILR',
        help='learning rate (default: 0.01)',
    )
    # 奖励的折扣因子
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        metavar='G',
        help='discount factor for rewards (default: 0.99)',
    )
    # GAE的参数
    parser.add_argument(
        '--tau',
        type=float,
        default=1.00,
        metavar='T',
        help='parameter for GAE (default: 1.00)',
    )
    # 熵正则化项
    parser.add_argument(
        '--beta',
        type=float,
        default=1e-2,
        help='entropy regularization term'
    )
    # dropout比率
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=0,
        help='The dropout ratio to use (default is no dropout).',
    )
    # LSTM隐藏状态的大小
    parser.add_argument(
        '--hidden-state-sz',
        type=int,
        default=512,
        help='size of hidden state of LSTM.'
    )
    # 注意力头的数量
    parser.add_argument(
        '--nhead',
        type=int,
        default=8,
    )
    # LSTM中的注意力头数量
    parser.add_argument(
        '--lstm-nhead',
        type=int,
        default=4,
    )
    # 编码器层的数量
    parser.add_argument(
        '--num-encoder-layers',
        type=int,
        default=6,
    )
    # 解码器层的数量
    parser.add_argument(
        '--num-decoder-layers',
        type=int,
        default=6,
    )
    # 前馈层的维度
    parser.add_argument(
        '--dim-feedforward',
        type=int,
        default=512,
    )
    # 模仿学习损失的比率
    parser.add_argument(
        '--il-rate',
        type=float,
        default=0.1,
        help='the rate of imitation learning loss'
    )
    # 是否在输入前进行动作嵌入
    parser.add_argument(
        '--action-embedding-before',
        action='store_true',
    )
    #  多头注意力的门控设置
    parser.add_argument(
        '--multihead-attn-gates',
        type=str,
        default=['input', 'cell'],
        nargs='+',
    )
    # 是否替换输入门
    parser.add_argument(
        '--replace-input-gate',
        action='store_true',
    )
    # 学习率下降的权重
    parser.add_argument(
        '--lr-drop-weight',
        type=float,
        default=0.1
    )
    # 学习率下降的eps
    parser.add_argument(
        '--lr-drop-eps',
        type=int,
        default=None,
    )
    # 最小学习率
    parser.add_argument(
        '--lr-min',
        type=float,
        default=0.0001,
    )
    
    # arguments related with pretraining Visual Transformer
    # 学习率下降的步数
    parser.add_argument(
        '--lr-drop',
        default=10,
        type=int
    )
    # 梯度裁剪的最大范数
    parser.add_argument(
        '--clip_max_norm',
        default=0.1,
        type=float,
        help='gradient clipping max norm'
    )
    # 打印频率
    parser.add_argument(
        '--print-freq',
        type=int,
        default=500,
    )

    # ==================================================
    # arguments related with DETR detector
    # 是否使用DETR检测器
    parser.add_argument(
        '--detr',
        action='store_true',
    )
    #  是否对DETR检测特征中的非对象类别进行填充
    parser.add_argument(
        '--detr-padding',
        action='store_false',
        help='padding non-object classes in detr detection featuers with 0'
    )

    args = parser.parse_args()

    return args
