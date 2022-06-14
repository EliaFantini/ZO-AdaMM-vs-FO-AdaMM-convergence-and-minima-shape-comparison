def print_filters(path_zo, path_fo,num_experiment, nb=10):

    conv_data = load_weights_sequence_all_layers(path_zo, path_fo, nb, conv_layers=True)

    conv_data.pop('conv1.bias')
    conv_data.pop('conv2.bias')
    conv_data.pop('conv2.weight')
    # Plot the convolutional layers


    for i, (l, d) in enumerate(conv_data.items()):
        fig, axs = plt.subplots(2, 3)
        print_weights(d['zo'][num_experiment], d['fo'][num_experiment], axs)

        fig = plt.gcf()
        fig.suptitle("First convolutional layer's filters", fontsize=16)
        fig.set_size_inches(20, 20)
        plt.show()


def print_weights(zo_data, fo_data, axs, num_channels = 3):
    nb_zo = zo_data.shape[0]
    X = np.zeros((nb_zo*2*num_channels,int(zo_data.shape[1]/num_channels)))
    for i in range(num_channels):
        X[i*nb_zo:(i+1)*nb_zo,:] = zo_data[:,i*X.shape[1]:(i+1)*X.shape[1]]
        X[(nb_zo*num_channels + i * nb_zo):(nb_zo*num_channels + (i + 1) * nb_zo), :] = fo_data[:,
                                                                i * X.shape[1]:(i + 1) * X.shape[1]]


    #fig, ax = plt.subplots()
    for i in range(num_channels):
        axs[0,i].imshow(X[(i+1)*nb_zo-1,:].reshape((int(math.sqrt(X.shape[1])),int(math.sqrt(X.shape[1])))), cmap='gray')
        axs[0, i].set_title(f"ZO filter {i}")
        axs[1, i].imshow(X[nb_zo*num_channels+(i+1)*nb_zo-1,:].reshape((int(math.sqrt(X.shape[1])),int(math.sqrt(X.shape[1])))), cmap='gray')
        axs[1, i].set_title(f"FO filter {i}")



def project_weights_per_filter(zo_data, fo_data, ax, title='', num_channels = 3):

    nb_zo = zo_data.shape[0]
    X = np.zeros((nb_zo*2*num_channels,int(zo_data.shape[1]/num_channels)))
    for i in range(num_channels):
        X[i*nb_zo:(i+1)*nb_zo,:] = zo_data[:,i*X.shape[1]:(i+1)*X.shape[1]]
        X[(nb_zo*num_channels + i * nb_zo):(nb_zo*num_channels + (i + 1) * nb_zo), :] = fo_data[:,
                                                                i * X.shape[1]:(i + 1) * X.shape[1]]
    X_emb = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42).fit_transform(X)

    X_emb_zo = X_emb[:nb_zo*num_channels , :]
    X_emb_fo = X_emb[nb_zo*num_channels:, :]

    #fig, ax = plt.subplots()
    for i in range(num_channels):
        if i==0:
            ax.plot(X_emb_zo[i*nb_zo:(i+1)*nb_zo, 0], X_emb_zo[i*nb_zo:(i+1)*nb_zo, 1],'b', marker='x', label=f'ZO')
            ax.plot(X_emb_fo[i*nb_zo:(i+1)*nb_zo, 0], X_emb_fo[i*nb_zo:(i+1)*nb_zo, 1],'y', marker='x', label=f'FO')
        else:
            ax.plot(X_emb_zo[i * nb_zo:(i + 1) * nb_zo, 0], X_emb_zo[i * nb_zo:(i + 1) * nb_zo, 1], 'b', marker='x')
            ax.plot(X_emb_fo[i * nb_zo:(i + 1) * nb_zo, 0], X_emb_fo[i * nb_zo:(i + 1) * nb_zo, 1], 'y', marker='x')

        ax.annotate('start', (X_emb_zo[i*nb_zo, 0] + 0.3, X_emb_zo[i*nb_zo, 1]))
        ax.annotate('end', (X_emb_zo[(i + 1)*nb_zo - 1, 0] + 0.3, X_emb_zo[(i + 1)*nb_zo - 1, 1]))
        ax.annotate('start', (X_emb_fo[i*nb_zo, 0] + 0.3, X_emb_fo[i*nb_zo, 1]))
        ax.annotate('end', (X_emb_fo[(i + 1)*nb_zo - 1, 0] + 0.3, X_emb_fo[(i + 1)*nb_zo - 1, 1]))
        ax.legend()

    fig = plt.gcf()
    #fig.legend()
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    ax.set_title(f'Layer : {title}')


def project_weights_all_layers_per_filter(path_zo, path_fo, experiment_num, nb=10):

    conv_data = load_weights_sequence_all_layers(path_zo, path_fo, nb, conv_layers=True)

    conv_data.pop('conv1.bias')
    conv_data.pop('conv2.bias')
    # Plot the convolutional layers
    fig, axs = plt.subplots(1, 2)

    for i, (l, d) in enumerate(conv_data.items()):
        project_weights_per_filter(d['zo'][experiment_num], d['fo'][experiment_num], axs[i], title=l)

    fig = plt.gcf()
    fig.set_size_inches(10, 3)
    fig.text(0.5, 0.01, 'x', ha='center')
    fig.text(0.1, 0.5, 'y', va='center', rotation='vertical')
    fig.suptitle(f"Experiment {experiment_num + 1}", fontsize=16)

    plt.show()
