An implementation of EPGN in pytorch
以WGAN为基础，使用了episode-based的训练方式。
![image](https://github.com/CallMeNumber1/EPGN-pytorch/blob/master/EPGN.png)
#### Intro
- 仅展示了在awa1上的结果：
- 运行的参数：epoch 30， episode 50，inner loop 100， bs 100
论文中报告的结果：ZSL(ACC):74.4 GZSL(H):71.2
> 以下模型选择是选择GZSL上的性能（即H）最高的作为标准
原版tensorflow实现实际运行结果：@epoch 23: ZSL:71.6 GZSL:71.2
此pytorch版本实际运行结果：@epoch 24: ZSL:70.9 GZSL:70.3
仿照tf进行权重初始化后的运行结果：@epoch 23: ZSL:73.4 GZSL:72.8，比论文中GZSL的性能还高了1.6个点。
#### File Intro
- log: log1.txt和log1_1.txt是固定了seed之后代码的两次运行，来确保运行结果一致。log_new_d.txt是将代码中第二次训练netG时，使用刚刚更新过的netD，这样效果反而变差。log_weight_init.txt是仿照tf中truncated_normal进行权重初始化，相比于默认的权重初始化，效果提高了许多（GZSL上提升2.5个点）。
#### TODO
- data load 改成pytorch标准的
- hyperparam logging script .sh
- tf有多个优化器，pytorch如何对应 [DONE]
- 论文中G和F是三层，实际代码中只有二层，且实际代码中没有dropout；之前大部分GAN用的是leaky relu，文中用的是relu，
- WGAN的实现方式不太一样，也可能是影响复现效果的原因
- 最后inference阶段，代码简化一下

