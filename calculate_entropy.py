# -- Imports -- #
from pitchfx_models import *
from datetime import datetime
import tabulate
import numpy
import progressbar
from scipy import stats
import peewee

# -- Constants -- #

# The PitchTypes table proved incomplete; these are all of the pitch types to appear in 2015 data,
# After removing pitches classified as NULL, an empty string, or unknown ('IN' or 'UN')
PITCH_TYPES_2015_DATA = ('FA', 'CH', 'SL', 'SI', 'CU', 'FS', 'FF', 'FC', 'PO', 'AB', 'KN', 'FT', 'KC', 'FO', 'EP', 'SC')


# -- Helper Methods -- #

def entropy_from_probabilities(probabilities):
    """
    H(X) = Sum over x in Ax: P(x) * log2(1/P(x))
    :param probabilities: 2d dict (dictionary of dictionaries)
    :return: The ensemble entropy for the probabilities given
    """

    joint_entropy = 0
    for p_x in probabilities:
        if p_x:
            joint_entropy += (p_x * numpy.log2(1.0 / p_x))

    return joint_entropy

    # return sum([p_x * numpy.log2(1.0 / p_x) for p_x in probabilities if p_x])


def get_single_pitch_probabilities(query_filter, start_date=datetime(2015, 1, 1)):
    """
    Calculate the independent probabilities of each pitch occurring
    :param query_filter: A filter for the query, so far used to query for a specific pitcher
    (see usage in print_for_pitcher)
    :param start_date: A start date, after I discovered I was querying over all pitches. Defaults to 01/01/2015.
    :return: A dict from each pitch type occurring to its independent probability of occurring
    """
    pitch_counts = {}

    progress = progressbar.ProgressBar()
    # for pitch_type in progress(PitchType.select()): # See comment near the definition of the constant for explanation
    for pitch_type in progress(PITCH_TYPES_2015_DATA):
        # current_type = pitch_type.id
        current_type = pitch_type

        count_query = Pitch.select(Pitch, AtBat, Game). \
            join(AtBat, on=(Pitch.ab_id == AtBat.ab)). \
            join(Game, on=(AtBat.game_id == Game.game_id)). \
            where(Pitch.pitch_type == current_type). \
            where(Game.date >= start_date)

        if query_filter:
            count_query = count_query.where(query_filter)

        # As we query over all non-null/empty string/unknown pitch types, ignore the zero count ones
        current_type_count = count_query.count()
        if current_type_count:
            pitch_counts[current_type] = current_type_count

    total = sum(pitch_counts.values())
    probabilities = {pt: (float(pitch_counts[pt]) / total) for pt in pitch_counts}

    return probabilities


def print_independent_probabilities(query_filter=None):
    """
    Print the independent probabilities - both as a 1d probability list (basically a PDF) or a 2d matrix of independent
    transitions - not a markov chain.
    :param query_filter: The query filter to query probabilities with. So far used to query for a specific player
    :return: Three return items:
        ordered_pitch_types - the pitch types seen in this sample, ordered by independent probability descending
        probabilities - the 1d dict of independent probabilities
        joint_probabilities - the 2d dict (dict of dicts) of joint probabilities under the assumption of independence
    """
    probabilities = get_single_pitch_probabilities(query_filter)

    ordered_pitch_types = sorted(probabilities.keys(), key=probabilities.get, reverse=True)

    # First printing just the individual probabilities
    print 'Independent pitch probabilities:'
    print tabulate.tabulate([[probabilities[pitch_type] for pitch_type in ordered_pitch_types]], ordered_pitch_types)
    print 'The entropy of the independent probabilities is {entropy}'.format(
        entropy=entropy_from_probabilities(probabilities.values()))

    # The rows are lists for printing; the joint_probabilities dict for returning and using later
    rows = []
    joint_probabilities = {}
    progress = progressbar.ProgressBar()

    for first_pitch_type in progress(ordered_pitch_types):
        row = [first_pitch_type]
        joint_probabilities[first_pitch_type] = {}

        for second_pitch_type in ordered_pitch_types:
            joint_probability = probabilities[first_pitch_type] * probabilities[second_pitch_type]
            row.append(joint_probability)
            joint_probabilities[first_pitch_type][second_pitch_type] = joint_probability

        rows.append(row)

    print 'Joint independent probabilities:'
    title = ['*****'] + ordered_pitch_types
    print tabulate.tabulate(rows, title)

    return ordered_pitch_types, probabilities, joint_probabilities


def query_transition_counts(ordered_pitch_types, query_filter):
    """
    Query the PitchFX database for the counts of transitions, using the pitch types found to exist when querying
    for pitches independently
    :param ordered_pitch_types: The pitch types encountered by the independent query
    :param query_filter: A filter to query by - so far used to query for an individual player
    :return: A 2d dict (dict of dicts) of the pitch transitions seen -  transition_counts[A][B]
    holds the number of transitions seen from A to B
    """
    transition_counts = {}

    for first_pitch_type in ordered_pitch_types:
        transition_counts[first_pitch_type] = {}

        for second_pitch_type in ordered_pitch_types:
            count_query = PitchTransition.select().where(
                PitchTransition.first_pitch_type == first_pitch_type).where(
                PitchTransition.second_pitch_type == second_pitch_type)

            if query_filter:
                count_query = count_query.where(query_filter)

            transition_counts[first_pitch_type][second_pitch_type] = count_query.count()

    return transition_counts


def joint_probabilities_from_transitions(ordered_pitch_types, transition_counts):
    """
    Calculate the joint probabilities from the transition counts
    :param ordered_pitch_types: he pitch types encountered by the independent query
    :param transition_counts: The counts of transitions (encountered in querying the DB or simulated in the permutation
    test)
    :return: Three return values:
        joint_probabilities - a 2d dict (dict of dicts) holding the transitions probabilities -
            joint_probabilities[A][B] holds the conditional probability of A and B - P(A & B) = P(A)*P(B|A)
        markov_rows - the 2d list (list of lists) of the rows of the markov matrix describing the transitions -
            P(B|A) for the row A and column B
        total_transitions - the total number of transitions seen in the the sample
    """
    first_pitch_totals = {first_pitch_type: sum(transition_counts[first_pitch_type].values())
                          for first_pitch_type in ordered_pitch_types}

    total_transitions = sum(first_pitch_totals.values())

    markov_rows = []
    joint_probabilities = {}

    for first_pitch_type in ordered_pitch_types:
        first_pitch_transitions = transition_counts[first_pitch_type]
        joint_probabilities[first_pitch_type] = {}
        first_pitch_type_probability = float(first_pitch_totals[first_pitch_type]) / total_transitions

        second_pitch_total = sum(first_pitch_transitions.values())
        row = [first_pitch_type]

        for second_pitch_type in ordered_pitch_types:
            if second_pitch_total == 0:
                second_pitch_conditional_probability = 0

            else:
                second_pitch_conditional_probability = \
                    float(first_pitch_transitions[second_pitch_type]) / second_pitch_total

            row.append(second_pitch_conditional_probability)

            joint_probabilities[first_pitch_type][second_pitch_type] = \
                first_pitch_type_probability * second_pitch_conditional_probability

        markov_rows.append(row)

    return joint_probabilities, markov_rows, total_transitions


def print_conditional_probabilities(ordered_pitch_types, query_filter=None):
    """
    Prints the conditional probability markov matrix
    :param ordered_pitch_types: The pitch types encountered by the independent query (ordered by individual counts desc)
    :param query_filter:  A filter to query by - so far used to query for an individual player
    :return: Two return values:
        joint_probabilities - a 2d dict (dict of dicts) holding the transitions probabilities -
            joint_probabilities[A][B] holds the conditional probability of A and B - P(A & B) = P(A)*P(B|A)
        total_transitions - the total number of transitions seen in the the sample
    """
    transition_counts = query_transition_counts(ordered_pitch_types, query_filter)

    joint_probabilities, markov_rows, total_transitions = joint_probabilities_from_transitions(ordered_pitch_types,
                                                                                               transition_counts)

    print 'Observed pitch transition Markov matrix:'
    title = ['*****'] + ordered_pitch_types
    print tabulate.tabulate(markov_rows, title)

    return joint_probabilities, total_transitions


def entropy_from_probability_matrix(matrix):
    """
    H(X,Y) = Sum over x in Ax, y in Ay: P(x,y) * log2(1/P(x,y))
    :param matrix: 2d dict (dictionary of dictionaries)
    :return: The ensemble entropy for the probability matrix given
    """
    joint_entropy = 0
    for x in matrix:
        current = matrix[x]

        for y in current:
            p_x_y = current[y]

            if p_x_y:  # make sure p_x_y != 0
                joint_entropy += (p_x_y * numpy.log2(1.0 / p_x_y))

    return joint_entropy


def kullback_leibler_divergence(p, q):
    """
    D_KL(P||Q) = Sum over over x: P(x) * log2(P(x) / Q(x))
    :param p: 2d dict (dictionary of dictionaries)
    :param q: 2d dict (dictionary of dictionaries) with the same structure (keys in each dict) as P
    :return: The Kullback_Leibler Divergence of the two distributions
    """
    joint_entropy = 0
    for x in p:
        p_x = p[x]
        q_x = q[x]

        for y in p_x:
            p_x_y = p_x[y]
            q_x_y = q_x[y]

            if p_x_y and q_x_y:
                joint_entropy += (p_x_y * numpy.log2(p_x_y / q_x_y))

    return joint_entropy


def entropy_permutation_test(ordered_pitch_types, single_pitch_pdf, conditional_joint_probabilities, total_transitions,
                             n=1000):
    """
    A permutation-test for the joint entropy. Taking the PDF for single pitches, generating ('n') permutations of the
    same length ('total_transitions') as the observed one, and calculating the entropy of each permutation. Then,
    doing a t-test using the permutation entropies as the sample and the observed conditional entropy as the test value.
    :param ordered_pitch_types: he pitch types encountered by the independent query (ordered by individual counts desc)
    :param single_pitch_pdf: A dictionary from individual pitch to its probability of occurring
    :param conditional_joint_probabilities: a 2d dict (dict of dicts) holding the transitions probabilities -
            conditional_joint_probabilities[A][B] holds the conditional probability of A and B - P(A & B) = P(A)*P(B|A)
    :param total_transitions: The total number of transitions observed - size of each permutation to generate
    :param n: The number of permutations to generate (defaults to 100)
    :return: The z-score and p-value resulting from the permutation test
    """
    pitch_types, pitch_probabilities = zip(*single_pitch_pdf.items())
    permutation_entropies = []
    progress = progressbar.ProgressBar()

    for test_number in progress(xrange(n)):
        # create the new matrix
        permutation_counts = {}
        for first_pitch_type in ordered_pitch_types:
            permutation_counts[first_pitch_type] = {}
            for second_pitch_type in ordered_pitch_types:
                permutation_counts[first_pitch_type][second_pitch_type] = 0

        pitch_permutation = numpy.random.choice(pitch_types, total_transitions, p=pitch_probabilities)
        current_pitch = numpy.random.choice(pitch_types, p=pitch_probabilities)
        for next_pitch in pitch_permutation:
            permutation_counts[current_pitch][next_pitch] += 1
            current_pitch = next_pitch

        joint_probabilities, _, _ = joint_probabilities_from_transitions(ordered_pitch_types, permutation_counts)
        permutation_entropies.append(entropy_from_probability_matrix(joint_probabilities))

    joint_entropy = entropy_from_probability_matrix(conditional_joint_probabilities)
    # print 'Mean', numpy.mean(permutation_entropies)
    # print 'Standard deviation', numpy.std(permutation_entropies)
    # tdof, tloc, tscale = stats.t.fit(permutation_entropies)
    # print 'DF', tdof, 'Loc (mean)', tloc, 'Scale (SD)', tscale
    # t_score = (joint_entropy - tloc) / tscale
    # print stats.t.cdf(joint_entropy, df=tdof, loc=tloc, scale=tscale)

    mean, stddev = stats.norm.fit(permutation_entropies)
    print 'Mean = {mean}\t StdDev = {stddev}'.format(mean=mean, stddev=stddev)
    z_score = (joint_entropy - mean) / stddev
    p_value = stats.norm.cdf(joint_entropy, mean, stddev)
    print 'The joint entropy has a Z-score of {z_score} which gives a P-value of {p_value}'.format(z_score=z_score,
                                                                                                   p_value=p_value)
    return z_score, p_value


def print_entropies(independent_joint_probabilities, conditional_joint_probabilities):
    """
    Print the entropies of the joint probabilities under the assumption of independence and the observed transitions
    :param independent_joint_probabilities: The joint probabilities under the assumption of independence -
        P(A & B) = P(A) * P(B|A) = P(A) * P(B)
    :param conditional_joint_probabilities: The joint probabilities from the transitions observed -
        P(A & B) = P(A) P (B|A)
    :return: The independent and conditional calculated entropies
    """
    indepndent_entropy = entropy_from_probability_matrix(independent_joint_probabilities)
    conditional_entropy = entropy_from_probability_matrix(conditional_joint_probabilities)

    print 'Independent H(X,Y) = {h}'.format(h=indepndent_entropy)
    print 'Conditional H(X,Y) = {h}'.format(h=conditional_entropy)
    print 'D_KL(Independent, Conditional) = {d_kl}' \
        .format(d_kl=kullback_leibler_divergence(independent_joint_probabilities, conditional_joint_probabilities))
    print 'D_KL(Conditional, Independent) = {d_kl}' \
        .format(d_kl=kullback_leibler_divergence(conditional_joint_probabilities, independent_joint_probabilities))

    return indepndent_entropy, conditional_entropy


def print_probabilities_and_entropies(independent_filter=None, conditional_filter=None):
    """
    Calculate the probabilities under the assumption of independence and the observed transition probabilities.
    Afterwards, calculate the entropies and print them, and perform the entropy permutation test.
    :param independent_filter: The filter to query the independent pitches (filtering the Pitch/AtBat objects)
    :param conditional_filter: The filter to query the pitch transitions (filter the PitchTransition object)
    :return: The independent and conditional entropies, and z-score and p-value resulting from the permutation test
    """
    ordered_pitch_types, single_pitch_pdf, independent_joint_probabilities = print_independent_probabilities(
        independent_filter)
    conditional_joint_probabilities, total_transitions = print_conditional_probabilities(ordered_pitch_types,
                                                                                         conditional_filter)

    independent_entropy, conditional_entropy = \
        print_entropies(independent_joint_probabilities, conditional_joint_probabilities)

    z_score, p_value = entropy_permutation_test(ordered_pitch_types, single_pitch_pdf,
                                                conditional_joint_probabilities, total_transitions)

    return independent_entropy, conditional_entropy, z_score, p_value


def print_probabilities_and_entropies_for_pitcher(pitcher_eliasid):
    """
    Same as print_probabilities_and_entropies, but for a specific pitcher (by his elias ID)
    :param pitcher_eliasid: The elias ID of the pitcher to test
    :return: Nothing - test results printed to stdout
    """
    pitcher = Player.get(Player.eliasid == pitcher_eliasid)
    print 'Printing for {first} {last}'.format(first=pitcher.first, last=pitcher.last)
    independent_entropy, conditional_entropy, z_score, p_value = \
        print_probabilities_and_entropies(AtBat.pitcher == pitcher, PitchTransition.pitcher == pitcher)

    return ('{first} {last}'.format(first=pitcher.first, last=pitcher.last), independent_entropy, conditional_entropy,
            z_score, p_value)


FINAL_TABLE_HEADERS = ('Name', 'Independent Entropy', 'Conditional Entropy', 'Z-Score', 'P-Value')
MEAN_TRANSITION_COUNT_GROUP = (408061, 554340, 489265, 502706, 570663, 453646, 543424, 451661, 433586, 458708)
MEAN_AND_SIGMA_TRANSITION_COUNT_GROUP = (606273, 408241, 547874, 518452, 448802, 554234, 592332, 547179, 608665, 218596)
MAX_TRANSITION_COUNT_GROUP = (543521, 500779, 502042, 430935, 477132, 572971, 456501, 456034, 450172, 453562)

UNIQUE_PITCHERS = (121250, 123801, 285079) # Mariano Rivera, Tim Wakefield, and R.A. Dickey

def unique_independent_probs():
    '''
    Print the independent probabilities only (as some of these pitchers did not pitch in 2015 for three unique pitchers:
    Mariano Rivera, famed for his cutter, and Tim Wakefield and R.A. Dickey, successful knuckleballers
    :return: Nothing; prints pitch probabilities to stdout
    '''
    for pitcher_eliasid in UNIQUE_PITCHERS:
        pitcher = Player.get(Player.eliasid == pitcher_eliasid)
        print 'Printing for {first} {last}'.format(first=pitcher.first, last=pitcher.last)

        probabilities = get_single_pitch_probabilities(AtBat.pitcher == pitcher, start_date=datetime(2007,1,1))
        ordered_pitch_types = sorted(probabilities.keys(), key=probabilities.get, reverse=True)

        print 'Independent pitch probabilities:'
        print tabulate.tabulate([[probabilities[pitch_type] for pitch_type in ordered_pitch_types]], ordered_pitch_types)


PITCHERS_ABOVE_MEAN_PT_QUERY = \
    'select * from players p where p.eliasid in ' + \
    '(select pt.pitcher from pitch_transitions pt group by pt.pitcher having count(pt.`pitcher`) > 688)'


def all_pitchers_above_mean_pt():
    pitchers = peewee.RawQuery(Player, PITCHERS_ABOVE_MEAN_PT_QUERY)
    rows = []

    for pitcher in pitchers:
        rows.append(print_probabilities_and_entropies_for_pitcher(pitcher.eliasid))

    print tabulate.tabulate(rows, FINAL_TABLE_HEADERS)

    print 'With {n} pitchers, the mean z-score is {mean}'.format(n=len(rows), mean=numpy.mean([r[3] for r in rows]))


def main():
    # all_pitchers_above_mean_pt()
    final_table_rows = []

    for pitcher_set in (MEAN_TRANSITION_COUNT_GROUP, MEAN_AND_SIGMA_TRANSITION_COUNT_GROUP,
                        MAX_TRANSITION_COUNT_GROUP):

        for pitcher_id in pitcher_set:
            final_table_rows.append(print_probabilities_and_entropies_for_pitcher(pitcher_id))

    print tabulate.tabulate(final_table_rows, FINAL_TABLE_HEADERS)

    print_probabilities_and_entropies()
    unique_independent_probs()


if __name__ == '__main__':
    main()

