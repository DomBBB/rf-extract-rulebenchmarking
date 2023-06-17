
RULESET = [lambda x: x[0] < 0.2 and x[2] > 0.7,
           lambda x: 0.5 < x[1] < 0.8,
           lambda x: x[2] < 0.1 and x[1] < 0.1]


def _generate_interval_rule(t1, t2):
    return lambda x: t1 <= x[0] <= t2


def generate_interval_rule_set(size, log_file):
    rules = []
    number_of_intervals = size
    interval_length = 1/(2*number_of_intervals)
    for i in range(number_of_intervals):
        t1 = 2*i*interval_length
        t2 = (2*i+1)*interval_length
        rule = _generate_interval_rule(t1, t2)
        rules.append(rule)
        _ = log_file.write(str(t1)+' <= x1 <= '+str(t2)+' v \n')
    return rules


def generate_linear_rule_set(upper_bound, log_file):
    _ = log_file.write('sum x[i], i=0, ...,'+str(upper_bound)+' <= '+str(upper_bound+1)+'/2')
    return [lambda x: sum([x[k] for k in range(upper_bound+1)]) <= (upper_bound+1)/2]


def generate_conjunctive_rule(upper_bound, log_file):
    _ = log_file.write('for all i=0, ...,'+str(upper_bound)+': x[i] <= 0.5')
    return [lambda x: all([x[k] <= 0.5 for k in range(upper_bound+1)])]


def generate_conjunctive_rule2(upper_bound, log_file):
    bound = 0.5 ** (1/(upper_bound+1))
    _ = log_file.write('for all i=0, ...,'+str(upper_bound)+': x[i] <= ' + str(bound))
    return [lambda x: all([x[k] <= bound for k in range(upper_bound+1)])]

