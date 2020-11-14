def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data.float())
        loss = loss_function(recon_batch, data.float(), mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
​
    print('====> Epoch: {} Average loss: {:.4f}'.format(




def loss_function(recon_x, x, mu, logvar):
    L1_reconstruction = F.l1_loss(recon_x, x.view(-1, 16320), reduction='sum')
​
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return L1_reconstruction, KLD


def lagrangian_loss(recon_x, x, mu, logvar, lambda_param, kappa):
    lam_x = lambda_param * x
    lam_recon_x = lambda_param * recon_x
    L1_loss = F.l1_loss(lam_recon_x, lam_x.view(-1, 16320), reduction='sum')
​    L1_constrained = L1_loss - (kappa**2)
    #If the above does not work, we might have to do something else like:


    constraint = torch.abs(recon_x + x) #then manipulate outside in GECO_train.
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD, L1_constrained

def GECO_train(epoch, parameters = {"t":0, "lambda":torch.ones(1765424),"cma":None, "cma_alpha":0.99}):
    model.train()
    train_loss = 0
    for batch_idx, (data) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data.float())
        KLD, constraint = lagrangian_loss(recon_batch, data.float(), mu, logvar)
        if parameters["t"] == 0:
            parameters["cma"] = constraint
        else:
            #this might need some tensorflow magic.
            parameters["cma"] = parameters["cma_alpha"]*parameters["cma"] + (1-parameters["cma_alpha"]) * constraint
        constraint = constraint + StopGradient(parameters["cma"] - constraint) #Need to set this tensor's requires_grad to False
        loss = KLD + parameters["lambda"] * constraint
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        #now update lambda
        parameters["lambda"] *= tf.exp(constraint)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))