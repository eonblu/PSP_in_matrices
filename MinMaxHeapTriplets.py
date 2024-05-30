# Triplets of [Value, Row, Column]
class MinMaxHeap(object):
	def __init__(self, reserve=0):
		self.a = [[None, None, None]] * reserve
		self.size = 0

	def __len__(self):
		return self.size

	def insert_triplet(self, triplet, CompsObj):
		"""
		Insert triplet into heap. Complexity: O(log(n))
		"""
		if len(self.a) < self.size + 1: # not a comparison of 2 values?
			self.a.append(triplet)
		insert_into_heap(self.a, triplet, self.size, CompsObj)
		self.size += 1

	def peekmin(self, CompsObj):
		"""
		Get minimum element. Complexity: O(1)
		"""
		return peekmin(self.a, self.size, CompsObj)

	def peekmax(self, CompsObj):
		"""
		Get maximum element. Complexity: O(1)
		"""
		return peekmax(self.a, self.size, CompsObj)

	def popmin(self, CompsObj):
		"""
		Remove and return minimum element. Complexity: O(log(n))
		"""
		m, self.size = removemin(self.a, self.size, CompsObj)
		return m

	def popmax(self, CompsObj):
		"""
		Remove and return maximum element. Complexity: O(log(n))
		"""
		m, self.size = removemax(self.a, self.size, CompsObj)
		return m


def level(i):
	return (i+1).bit_length() - 1


def trickledown(array, i, size, CompsObj):
	if level(i) % 2 == 0:  # min level # Not a comparison of 2 values?
		trickledownmin(array, i, size, CompsObj)
	else:
		trickledownmax(array, i, size, CompsObj)


def trickledownmin(array, i, size, CompsObj):
	if size > i * 2 + 1:  # i has children
		m = i * 2 + 1
		CompsObj.increment()
		if i * 2 + 2 < size and array[i*2+2][0] < array[m][0]:
			m = i*2+2
		child = True
		for j in range(i*4+3, min(i*4+7, size)):
			CompsObj.increment()
			if array[j][0] < array[m][0]:
				m = j
				child = False

		if child:
			CompsObj.increment()
			if array[m][0] < array[i][0]:
				array[i], array[m] = array[m], array[i]
		else:
			CompsObj.increment()
			if array[m][0] < array[i][0]:
				array[m], array[i] = array[i], array[m]
				CompsObj.increment()
				if array[m][0] > array[(m-1) // 2][0]:
					array[m], array[(m-1)//2] = array[(m-1)//2], array[m]
				trickledownmin(array, m, size, CompsObj)


def trickledownmax(array, i, size, CompsObj):
	if size > i * 2 + 1:  # i has children
		m = i * 2 + 1
		CompsObj.increment()
		if i * 2 + 2 < size and array[i*2+2][0] > array[m][0]:
			m = i*2+2
		child = True
		for j in range(i*4+3, min(i*4+7, size)):
			CompsObj.increment()
			if array[j][0] > array[m][0]:
				m = j
				child = False

		if child:
			CompsObj.increment()
			if array[m][0] > array[i][0]:
				array[i], array[m] = array[m], array[i]
		else:
			CompsObj.increment()
			if array[m][0] > array[i][0]:
				array[m], array[i] = array[i], array[m]
				CompsObj.increment()
				if array[m][0] < array[(m-1) // 2][0]:
					array[m], array[(m-1)//2] = array[(m-1)//2], array[m]
				trickledownmax(array, m, size, CompsObj)


def bubbleup(array, i, CompsObj):
	if level(i) % 2 == 0:  # min level
		CompsObj.increment()
		if i > 0 and array[i][0] > array[(i-1) // 2][0]:
			array[i], array[(i-1) // 2] = array[(i-1)//2], array[i]
			bubbleupmax(array, (i-1)//2, CompsObj)
		else:
			bubbleupmin(array, i, CompsObj)
	else:  # max level
		CompsObj.increment()
		if i > 0 and array[i][0] < array[(i-1) // 2][0]:
			array[i], array[(i-1) // 2] = array[(i-1) // 2], array[i]
			bubbleupmin(array, (i-1)//2, CompsObj)
		else:
			bubbleupmax(array, i, CompsObj)


def bubbleupmin(array, i, CompsObj):
	while i > 2:
		CompsObj.increment()
		if array[i][0] < array[(i-3) // 4][0]:
			array[i], array[(i-3) // 4] = array[(i-3) // 4], array[i]
			i = (i-3) // 4
		else:
			return


def bubbleupmax(array, i, CompsObj):
	while i > 2:
		CompsObj.increment()
		if array[i][0] > array[(i-3) // 4][0]:
			array[i], array[(i-3) // 4] = array[(i-3) // 4], array[i]
			i = (i-3) // 4
		else:
			return


def peekmin(array, size, CompsObj):
	assert size > 0
	return array[0]


def peekmax(array, size, CompsObj):
	assert size > 0
	if size == 1:
		return array[0]
	elif size == 2:
		return array[1]
	else:
		CompsObj.increment()
		if array[1][0] > array[2][0]:
			return array[1]
		else:
			return array[2]


def removemin(array, size, CompsObj):
	assert size > 0
	elem = array[0]
	array[0] = array[size-1]
	# array = array[:-1]
	trickledown(array, 0, size - 1, CompsObj)
	array.pop() # Added to remove last element in list
	return elem, size-1


def removemax(array, size, CompsObj):
	assert size > 0
	if size == 1:
		final_elem = array[0]
		array.pop() # Added to remove last element in list
		return final_elem, size - 1
	elif size == 2:
		final_max = array[1]
		array.pop() # Added to remove last element in list
		return final_max, size - 1
	else:
		CompsObj.increment()
		i = 1 if array[1][0] > array[2][0] else 2
		elem = array[i]
		array[i] = array[size-1]
		# array = array[:-1]
		trickledown(array, i, size - 1, CompsObj)
		array.pop() # Added to remove last element in list
		return elem, size-1


def insert_into_heap(array, k, size, CompsObj):
	array[size] = k
	bubbleup(array, size, CompsObj)


def minmaxheapproperty(array, size):
	for i, k in enumerate(array[:size]):
		if level(i) % 2 == 0:  # min level
			# check children to be larger
			for j in range(2 * i + 1, min(2 * i + 3, size)):
				if array[j][0] < k[0]:
					print(array, j, i, array[j], array[i], level(i))
					return False
			# check grand children to be larger
			for j in range(4 * i + 3, min(4 * i + 7, size)):
				if array[j][0] < k[0]:
					print(array, j, i, array[j], array[i], level(i))
					return False
		else:
			# check children to be smaller
			for j in range(2 * i + 1, min(2 * i + 3, size)):
				if array[j][0] > k[0]:
					print(array, j, i, array[j], array[i], level(i))
					return False
			# check grand children to be smaller
			for j in range(4 * i + 3, min(4 * i + 7, size)):
				if array[j][0] > k[0]:
					print(array, j, i, array[j], array[i], level(i))
					return False

	return True

def test(n):
	from random import randint
	a = [[-1, 0, 0]] * n
	l = []
	size = 0
	for i in range(n):
		x = randint(0, 5 * n)
		insert_into_heap(a, [x, i, i], size)
		size += 1
		l.append(x)
		assert minmaxheapproperty(a, size)

	assert size == len(l)

	while size > 0:
		assert min(l) == peekmin(a, size)[0]
		assert max(l) == peekmax(a, size)[0]
		if randint(0, 1):
			e, size = removemin(a, size)
			assert e[0] == min(l)
		else:
			e, size = removemax(a, size)
			assert e[0] == max(l)
		l[l.index(e[0])] = l[-1]
		l.pop(-1)
		assert len(a[:size]) == len(l)
		assert minmaxheapproperty(a, size)

	print("OK")

def test_heap(n):
	from random import randint
	heap = MinMaxHeap(n)
	l = []
	for i in range(n):
		x = randint(0, 5 * n)
		heap.insert_triplet([x, i, i])
		l.append(x)
		print (heap.a)
		assert minmaxheapproperty(heap.a, len(heap))

	assert len(heap) == len(l)
	print(heap.a)

	while len(heap) > 0:
		assert min(l) == heap.peekmin()[0]
		assert max(l) == heap.peekmax()[0]
		if randint(0, 1):
			print ("pop min")
			e = heap.popmin()[0]
			assert e == min(l)
		else:
			print ("pop max")
			e = heap.popmax()[0]
			assert e == max(l)
		l[l.index(e)] = l[-1]
		l.pop(-1)
		assert len(heap) == len(l)
		assert minmaxheapproperty(heap.a, len(heap))
		print(heap.a, heap.size)
	print("OK")

if __name__ == '__main__':
    test(15)