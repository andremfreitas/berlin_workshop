import schnetpack as spk
import schnetpack.transform as trn
import torch
import torchmetrics

custom_data = spk.data.AtomsDataModule(
    './ethanol_data.db',
    batch_size=10,
    distance_unit='Ang',
    property_units={'energy':'kcal/mol', 'forces':'kcal/mol/Ang'},
    num_train=1000,
    num_val=100,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),   
        trn.CastTo32()
    ],
    num_workers=19,
    pin_memory=False, # set to false, when not using a GPU
)
custom_data.prepare_data()
custom_data.setup()

# ^ had to remove this line trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=True), form the transformations
# in order to make it work.

###

# setup the model.

cutoff = 5.
n_atom_basis = 128

pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_interactions=3,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)
pred_U0 = spk.atomistic.Atomwise(n_in=n_atom_basis,  output_key='energy')

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_U0],
    postprocessors=[trn.CastTo64()]     # had to remove "trn.AddOffsets(QM9.U0, add_mean=True, add_atomrefs=True)"
)

output_U0 = spk.task.ModelOutput(
    name='energy',  # changed this because initially it was output.U0 -- I dont think in our case this -- in our case is "energy" 
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_U0],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)


import pytorch_lightning as pl
import os

saving_dir = 'results'

if not os.path.exists('results'):
    os.makedirs(saving_dir)

logger = pl.loggers.TensorBoardLogger(save_dir=saving_dir)
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(saving_dir, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=saving_dir,
    max_epochs=3, # for testing, we restrict the number of epochs
)
trainer.fit(task, datamodule=custom_data)