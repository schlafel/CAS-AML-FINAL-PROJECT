from torch.utils.tensorboard import SummaryWriter,summary
import numpy as np

def write_hp(writer,hparam_dict,metric_dict,epoch = 0):
    exp, ssi, sei = summary.hparams(hparam_dict, metric_dict, hparam_domain_discrete=None)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    for k, v in metric_dict.items():
        if v is not None:
            writer.add_scalar(k, v,global_step=epoch)

    return writer


for run in range(5):
    writer = SummaryWriter(log_dir=r"C:\Users\fs.GUNDP\Python\CAS-AML-FINAL-PROJECT/tmp/tb_logs/test"+str(run))
    hparam_dict = {"a": 2.0, "b": 1.5,"c":5,"lr":10}
    metric_dict = {"hparam/test_accuracy": 0.0,
                   "hparam/val_accuracy":0.0,
                   "hparam/train_accuracy":0.0}



    # write_hp(writer,hparam_dict,metric_dict)

    # writer.add_hparams({"a": 2.0, "b": 1.5}, {"hparam/test_accuracy": 0.7,
    #                                           "hparam/val_accuracy":0.5},run_name="test")
    acc = 0
    loss = 5
    #training
    for i in range(10):
        loss  -= np.random.random(1)/10
        acc  += np.random.random(1)/10


        metric_dict = {
            # "hparam/test_accuracy": acc+np.random.random(1)/10.,
                       "accuracy/Val": acc,
                       "accuracy/Train": acc+0.01,
                       "accuracy/Test": None,
                       "loss/Train": loss ,
                       "loss/Val": loss-0.01,
                       "loss/Test": None,

        }
        writer = write_hp(writer, hparam_dict, metric_dict,epoch=i)

    metric_dict = {"accuracy/Test": acc+np.random.random(1)/10.,
                   "loss/Test": loss+np.random.random(1)/10.}
    writer = write_hp(writer, hparam_dict, metric_dict,epoch=i)


    writer.close()
