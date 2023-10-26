def compute_cum_prob(counts, alpha):
    probs = np.array(counts)
    norm = probs.sum() + alpha
    probs = probs/norm
    probs = list(probs)
    probs.append(alpha/norm)

    cum = np.zeros(len(probs))
    for i in range(len(probs)):
        cum[i] = np.sum(probs[:i+1])
    return cum

def new_customer(counts, alpha):
    unif = stats.uniform()
    u = unif.rvs(1)
    cum = compute_cum_prob(counts, alpha)
    for i, prob_c in enumerate(cum):
        if u < prob_c:
            if i == len(cum)-1:
                counts.append(1)
            else:
                counts[i] +=1
            break
    return counts

def chinese_restaurant_process(counts, alpha, n_cust):
    for j in range(n_cust):
        counts = new_customer(counts, alpha)
    return counts

# Plot parameters
total_customers = 10000
samples_crp = 10

for alpha in [1, 10, 100]:
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    for j in range(samples_crp):
        counts = list()
        num_tables = list()
        for i in range(total_customers):
            counts = chinese_restaurant_process(counts, alpha, n_cust=1)
            num_tables.append(len(counts))
            print('Processed customer: Alpha %s Sample %s %s/%s' % (alpha, j, i+1, total_customers))
            clear_output(wait=True)
        if j != samples_crp-1:
            ax.plot(range(total_customers), num_tables)
    ax.plot(range(total_customers), num_tables, label='Sample of CRP')
    ax.set_title(r'Samples of CRP tables/cust $\alpha=%s$' % alpha, size=title_size)
    ax.set_ylabel('Number of tables', size=axis_size)
    ax.set_xlabel('Number of customers', size=axis_size)
    ax.tick_params(labelsize=axis_size-10)
    plt.legend(prop={'size': legend_size})
    plt.tight_layout()
    plt.savefig('images/crp/chinese_restaurant_process_tables_alpha_%s.png' % alpha, dpi=100)