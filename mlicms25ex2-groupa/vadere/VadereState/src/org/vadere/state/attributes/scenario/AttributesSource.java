package org.vadere.state.attributes.scenario;

import com.fasterxml.jackson.annotation.JsonView;
import org.vadere.annotation.helptext.HelpIgnore;
import org.vadere.state.attributes.spawner.AttributesRegularSpawner;
import org.vadere.state.attributes.spawner.AttributesSpawner;
import org.vadere.state.util.Views;
import org.vadere.util.geometry.shapes.VShape;
import org.vadere.util.reflection.VadereAttribute;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A source is a {@link ScenarioElement} which spawns agents.
 * The spawning behaviour is controlled by a {@link AttributesSpawner}.
 */
public class AttributesSource extends AttributesVisualElement {
	@Deprecated
	@VadereAttribute(exclude = true)
	@HelpIgnore
	public static final String CONSTANT_DISTRIBUTION = "constant";

	/**
	 * This list stores all target ids where agents spawned at this source can travel to.
	 */
	@JsonView(Views.CacheViewExclude.class) // ignore when determining if floor field cache is valid
	private List<Integer> targetIds = new ArrayList<>();

	/**
	 * The spawner attributes define the behaviour of the source.
	 * The possible spawners are:
	 * <ul>
	 *     <li><b>RegularSpawner</b> A spawner with a single distribution for sampling spawn events</li>
	 *     <li><b>LERPSpawner</b> </li>
	 *     <li><b>TimeseriesSpawner</b> A spawner which stores the number of elements to spawn each interval</li>
	 *     <li><b>MixedSpawner</b> A spawner which switches between distributions at given switchpoints</li>
	 * </ul>
	 */
	@JsonView(Views.CacheViewExclude.class)
	private AttributesSpawner spawner = new AttributesRegularSpawner();
	/**
	 *  This Attribute only takes affect if a model org.vadere.simulator.models.groups.GroupModel<br>
	 *  is present in the scenario. When this is the case this list defines the group size<br>
	 *  distribution of this source. The list can be arbitrary long but must add up to 1.<br>
	 *  The index of the list represents the size of the  groups and the value the probability<br>
	 *  index 0 => GroupSize = 1<br>
	 *  index 1 => GroupSize = 2<br>
	 *  ...
	 *
	 *  Example:<br>
	 *  probability [ 0.0, 0.0, 0.25, 0.25, 0.25, .... ]<br>
  	 *  GroupSize   [  1    2    3     4     5         ]<br>
	 *  uniform distribution of groups of the size from 3 to 5<br>
	 */

	@JsonView(Views.CacheViewExclude.class) // ignore when determining if floor field cache is valid
	private List<Double> groupSizeDistribution = new ArrayList<>( List.of(1.0));

	public AttributesSource() {
		super();
	}

	public AttributesSource(int id) {
		super();
		this.id = id;
	}

	public AttributesSource(int id, VShape shape) {
		super();
		this.id = id;
		this.shape = shape;
	}
//TODO attributesshape
	public AttributesSource(int id, VShape shape,AttributesSpawner spawner) {
		super();
		this.id = id;
		this.shape = shape;
		this.spawner = spawner;
	}

	public AttributesSpawner getSpawnerAttributes(){
		return this.spawner;
	}

	public void setSpawnerAttributes(AttributesSpawner spawner){
		this.spawner = spawner;
	}

	public List<Integer> getTargetIds() {
		return targetIds;
	}

	public List<Double> getGroupSizeDistribution() {
		return groupSizeDistribution;
	}

	public void setGroupSizeDistribution(ArrayList<Double> groupSizeDistribution) {
		checkSealed();
		this.groupSizeDistribution = groupSizeDistribution;
	}


	public void setTargetIds(ArrayList<Integer> targetIds) {
		checkSealed();
		this.targetIds = targetIds;
	}

	@Override
	public void check() throws IOException {
		try {
			/*VadereDistribution<?> distribution = DistributionFactory.create(
					this.getInterSpawnTimeDistribution(),
					this.getDistributionParameters(),
					this.getSpawnNumber(),
					new JDKRandomGenerator(42)
			);*/
		} catch (Exception e) {
			//throw new IOException("Cannot build " + this.getInterSpawnTimeDistribution());
		}

	}

	@Override
	public String toString() {
		return "AttributesSource{" +
				"spawner=" + spawner +
				", targetIds=" + targetIds +
				", groupSizeDistribution=" + groupSizeDistribution +
				", shape=" + shape +
				", visible=" + visible +
				", id=" + id +
				'}';
	}
}
