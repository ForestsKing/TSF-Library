# Long-term Time Series Forecasting Library (LTSFlib)

LTSFlib is an open-source library for long-term time series forecasting task.

To reflect the performance visually and realistically, we tested each algorithm fairly with uniform data input and hardware platform. We ran each subtask for 3 iters and took the average value as the final result. 

The detailed results are as follows:

<table>
    <tr>
        <th rowspan="2" style="text-align:center"> Model </th>
        <th colspan="2" style="text-align:center"> ETTh1 </th>
        <th colspan="2" style="text-align:center"> ETTh2 </th>
        <th colspan="2" style="text-align:center"> ETTm1 </th>
        <th colspan="2" style="text-align:center"> ETTm2 </th>
        <th rowspan="2" style="text-align:center"> Paper </th>
        <th rowspan="2" style="text-align:center"> Year </th>
    </tr>
    <tr>
        <th colspan="1" style="text-align:center">MSE</th>
        <th colspan="1" style="text-align:center">MAE</th>
        <th colspan="1" style="text-align:center">MSE</th>
        <th colspan="1" style="text-align:center">MAE</th>
        <th colspan="1" style="text-align:center">MSE</th>
        <th colspan="1" style="text-align:center">MAE</th>
        <th colspan="1" style="text-align:center">MSE</th>
        <th colspan="1" style="text-align:center">MAE</th>
    </tr>
    <!-- 模版
    <tr>
        <td style="text-align:center"> Model </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> <a href="">Link</a> </td>
        <td style="text-align:center"> 2023 </td> 	
    </tr>
    -->
    <tr>
        <td style="text-align:center"> Reformer </td>
        <td style="text-align:center"> 0.8234 </td>
        <td style="text-align:center"> 0.6690 </td>
        <td style="text-align:center"> 1.6758 </td>
        <td style="text-align:center"> 1.0473 </td>
        <td style="text-align:center"> 0.8355 </td>
        <td style="text-align:center"> 0.6485 </td>
        <td style="text-align:center"> 0.7868 </td>
        <td style="text-align:center"> 0.6837 </td>
        <td style="text-align:center"> <a href="https://arxiv.org/abs/2001.04451">Link</a> </td>
        <td style="text-align:center"> 2020 </td> 	
    </tr>
    <tr>
        <td style="text-align:center"> Informer </td>
        <td style="text-align:center"> 0.9678 </td>
        <td style="text-align:center"> 0.7704 </td>
        <td style="text-align:center"> 3.1124 </td>
        <td style="text-align:center"> 1.3865 </td>
        <td style="text-align:center"> 0.5769 </td>
        <td style="text-align:center"> 0.5348 </td>
        <td style="text-align:center"> 0.4520 </td>
        <td style="text-align:center"> 0.5308 </td>
        <td style="text-align:center"> <a href="https://ojs.aaai.org/index.php/AAAI/article/view/17325">Link</a> </td>
        <td style="text-align:center"> 2021 </td> 	
    </tr>
    <tr>
        <td style="text-align:center"> Autoformer </td>
        <td style="text-align:center"> 0.4578 </td>
        <td style="text-align:center"> 0.4566 </td>
        <td style="text-align:center"> 0.3998 </td>
        <td style="text-align:center"> 0.4286 </td>
        <td style="text-align:center"> 0.5128 </td>
        <td style="text-align:center"> 0.4780 </td>
        <td style="text-align:center"> 0.3109 </td>
        <td style="text-align:center"> 0.3515 </td>
        <td style="text-align:center"> <a href="https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html">Link</a> </td>
        <td style="text-align:center"> 2021 </td> 	
    </tr>
    <tr>
        <td style="text-align:center"> FEDformer </td>
        <td style="text-align:center"> 0.3887 </td>
        <td style="text-align:center"> 0.4248 </td>
        <td style="text-align:center"> 0.3525 </td>
        <td style="text-align:center"> 0.3968 </td>
        <td style="text-align:center"> 0.3977 </td>
        <td style="text-align:center"> 0.4278 </td>
        <td style="text-align:center"> 0.2082 </td>
        <td style="text-align:center"> 0.2932 </td>
        <td style="text-align:center"> <a href="https://proceedings.mlr.press/v162/zhou22g.html">Link</a> </td>
        <td style="text-align:center"> 2022 </td> 	
    </tr>
    <tr>
        <td style="text-align:center"> DLinear </td>
        <td style="text-align:center"> <b>0.3863</b> </td>
        <td style="text-align:center"> 0.4007 </td>
        <td style="text-align:center"> 0.3140 </td>
        <td style="text-align:center"> 0.3711 </td>
        <td style="text-align:center"> 0.3426 </td>
        <td style="text-align:center"> 0.3710 </td>
        <td style="text-align:center"> 0.1914 </td>
        <td style="text-align:center"> 0.2879 </td>
        <td style="text-align:center"> <a href="https://arxiv.org/abs/2205.13504">Link</a> </td>
        <td style="text-align:center"> 2022 </td>    
    </tr>
    <tr>
        <td style="text-align:center"> NLinear </td>
        <td style="text-align:center"> 0.3924 </td>
        <td style="text-align:center"> <b>0.4001</b> </td>
        <td style="text-align:center"> <b>0.2888</b> </td>
        <td style="text-align:center"> <b>0.3375</b> </td>
        <td style="text-align:center"> 0.3486 </td>
        <td style="text-align:center"> <b>0.3695</b> </td>
        <td style="text-align:center"> <b>0.1814</b> </td>
        <td style="text-align:center"> <b>0.2630</b> </td>
        <td style="text-align:center"> <a href="https://arxiv.org/abs/2205.13504">Link</a> </td>
	    <td style="text-align:center"> 2022 </td> 
    </tr>
    <tr>
        <td style="text-align:center"> TimesNet </td>
        <td style="text-align:center"> 0.4077 </td>
        <td style="text-align:center"> 0.4218 </td>
        <td style="text-align:center"> 0.3263 </td>
        <td style="text-align:center"> 0.3688 </td>
        <td style="text-align:center"> <b>0.3359</b> </td>
        <td style="text-align:center"> 0.3761 </td>
        <td style="text-align:center"> 0.1844 </td>
        <td style="text-align:center"> 0.2643 </td>
        <td style="text-align:center"> <a href="https://arxiv.org/abs/2210.02186">Link</a> </td>
        <td style="text-align:center"> 2022 </td> 	
    </tr>
</table>

## Develop your own model

- Add your model files to the folder `./layers/yourmodel/` and`./models/yourmodel.py`
- Include the newly added model in the `Exp_Main.model_dict` of `./exp/exp_main.py`.
- Create the corresponding scripts under the folder `./scripts/yourmodel.sh`.

## Contact

If you have any questions or suggestions, feel free to contact:

- Chengsen Wang ([cswang@bupt.edu.cn](mailto:cswang@bupt.edu.cn))
- Jinming Wu ([kimor.wu@outlook.com](mailto:kimor.wu@outlook.com))

or describe it in Issues.

## Acknowledgement

This library is constructed based on the following repos:

- Reformer: https://github.com/lucidrains/reformer-pytorch
- Informer: https://github.com/zhouhaoyi/Informer2020
- Autoformer: https://github.com/thuml/Autoformer
- FEDformer: https://github.com/MAZiqing/FEDformer
- DLinear: https://github.com/cure-lab/LTSF-Linear
- NLinear: https://github.com/cure-lab/LTSF-Linear
- TimesNet: https://github.com/thuml/TimesNet
