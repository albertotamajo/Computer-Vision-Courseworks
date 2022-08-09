package uk.ac.soton.ecs.at2n19.coursework3;

/**
 * Represent a pair
 * @param <K> class of the first value
 * @param <M> class of the second value
 */
public class Pair<K,M> {
    K val1;
    M val2;

    /**
     * Construct a pair with the given first and second value
     * @param val1 first value
     * @param val2 second value
     */
    public Pair(K val1, M val2) {
        this.val1 = val1;
        this.val2 = val2;
    }
}
