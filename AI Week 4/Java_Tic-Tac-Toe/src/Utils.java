import java.util.List;

/**
 * Some utility methods.
 * 
 * @author Dennis Soemers
 */
public class Utils {
	
	/**
	 * Removes element at given index. Does not shift all subsequent elements,
	 * but only swaps the last element into the removed index.
	 * @param <E>
	 * @param list
	 * @param idx
	 */
	public static <E> void removeSwap(final List<E> list, final int idx) {
		final int lastIdx = list.size() - 1;
		list.set(idx, list.get(lastIdx));
		list.remove(lastIdx);
	}

}
