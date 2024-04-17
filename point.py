from MiniProjectPath1 import dataset_1_filteredArray


class Point:
    def __init__(self, coords):
        """Inits a Point with a list of coordinates."""
        self.coords = coords
        self.currCluster = None

    @property
    def dim(self):
        return len(self.coords)

    def distFrom(self, other):
        """Calculates distance between two Points.

        Args:
          other: The Point we are calculating the distance from.

        Returns:
          A float representing the Euclidean distance between this point and other.
        """
        # Error checking, keep this here.
        if self.dim != other.dim:
            raise ValueError(
                "dimension mismatch: self has dim {} and other has dim {}".format(
                    self.dim, other.dim
                )
            )

        # Hint: Refer to the formula given in README.md for the Euclidean distance

        # fill in
        a = np.subtract(self.coords, other.coords)
        squared_dist = np.sum((a)**2, axis=0)
        dist = np.sqrt(squared_dist)

        return dist

    def moveToCluster(self, dest):
        """Reassigns this Point to a new Cluster.

        Args:
          dest: A Cluster object the Point will move to.
        
        Returns:
          True if dest is different from the current cluster, False otherwise.
        """
        if self.currCluster is dest:
            return False
        else:
            if self.currCluster:
                self.currCluster.removePoint(self)
            dest.addPoint(self)
            self.currCluster = dest
            return True

    def closest(self, objects):
        """Return the object that is closest to this point.

        Args:
          objects: A list of objects.

        Returns:
          The object in objects that is closest to this point. This
          object can be a Cluster or a Point.
        """
        minDist = self.distFrom(objects[0])
        minPt = objects[0]
        for p in objects:
            if self.distFrom(p) < minDist:
                minDist = self.distFrom(p)
                minPt = p
        return minPt

    def __getitem__(self, i):
        """p[i] will get the ith coordinate of the Point p."""
        return self.coords[i]

    def __str__(self):
        return str(self.coords)

    def __repr__(self):
        return f"Point({self.__str__()})"


def makePointList(data):
    """Creates a list of points from initialization data.
    #This function is outside Point Class.
    Args:
      data: A p-by-d numpy array.

    Returns:
      A list of length p containing d-dimensional Point objects, each Point's
      coordinates correspond to one row of data.
    """
    list = []
    # fill in
    for x in data:
        a = Point(x.tolist())
        list.append(a)


    return list


if __name__ == "__main__":
    data = dataset_1_filteredArray

    points = makePointList(data)
    print(points)