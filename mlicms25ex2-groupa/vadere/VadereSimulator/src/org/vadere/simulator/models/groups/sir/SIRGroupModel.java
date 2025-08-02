package org.vadere.simulator.models.groups.sir;


import org.vadere.simulator.context.VadereContext;
import org.vadere.annotation.factories.models.ModelClass;
import org.vadere.simulator.models.Model;
import org.vadere.simulator.models.groups.AbstractGroupModel;
import org.vadere.simulator.models.groups.Group;
import org.vadere.simulator.models.groups.GroupSizeDeterminator;
import org.vadere.simulator.models.groups.cgm.CentroidGroup;
import org.vadere.simulator.models.potential.fields.IPotentialFieldTarget;
import org.vadere.simulator.projects.Domain;
import org.vadere.state.attributes.Attributes;
import org.vadere.simulator.models.groups.sir.SIRGroup;
import org.vadere.state.attributes.models.AttributesSIRG;
import org.vadere.state.attributes.scenario.AttributesAgent;
import org.vadere.state.scenario.DynamicElementContainer;
import org.vadere.state.scenario.Pedestrian;
import org.vadere.state.scenario.Topography;
import org.vadere.state.types.PedestrianAttitudeType;
import org.vadere.util.geometry.LinkedCellsGrid;
import org.vadere.util.geometry.shapes.VRectangle;

import java.awt.geom.Rectangle2D;
import java.util.*;

/**
 * Implementation of groups for a susceptible / infected / removed (SIR) model.
 */
@ModelClass
public class SIRGroupModel extends AbstractGroupModel<SIRGroup> {

	protected Random random;
	private LinkedHashMap<Integer, SIRGroup> groupsById;
	private Map<Integer, LinkedList<SIRGroup>> sourceNextGroups;
	protected AttributesSIRG attributesSIRG;
	private Topography topography;
	private IPotentialFieldTarget potentialFieldTarget;
	private int totalInfected = 0;
	protected double simTimeStepLength;
	protected static final String simStepLength = "simTimeStepLength";
	// New attribute to track the latest simulation time for decoupling
	private double lastSimTime = 0.0;

	public SIRGroupModel() {
		this.groupsById = new LinkedHashMap<>();
		this.sourceNextGroups = new HashMap<>();
	}

	@Override
	public void initialize(List<Attributes> attributesList, Domain domain,
	                       AttributesAgent attributesPedestrian, Random random) {
		this.attributesSIRG = Model.findAttributes(attributesList, AttributesSIRG.class);
		this.topography = domain.getTopography();
		this.random = random;
        this.totalInfected = 0;
		this.simTimeStepLength = VadereContext.getCtx(this.topography).getDouble(simStepLength);
	}

	@Override
	public void setPotentialFieldTarget(IPotentialFieldTarget potentialFieldTarget) {
		this.potentialFieldTarget = potentialFieldTarget;
		// update all existing groups
		for (SIRGroup group : groupsById.values()) {
			group.setPotentialFieldTarget(potentialFieldTarget);
		}
	}

	@Override
	public IPotentialFieldTarget getPotentialFieldTarget() {
		return potentialFieldTarget;
	}

	private int getFreeGroupId() {
		if(this.random.nextDouble() < this.attributesSIRG.getInfectionRate()
        || this.totalInfected < this.attributesSIRG.getInfectionsAtStart()) {
			if(!getGroupsById().containsKey(SIRType.ID_INFECTED.ordinal()))
			{
				SIRGroup g = getNewGroup(SIRType.ID_INFECTED.ordinal(), Integer.MAX_VALUE/2);
				getGroupsById().put(SIRType.ID_INFECTED.ordinal(), g);
			}
            this.totalInfected += 1;
			return SIRType.ID_INFECTED.ordinal();
		}
		else{
			if(!getGroupsById().containsKey(SIRType.ID_SUSCEPTIBLE.ordinal()))
			{
				SIRGroup g = getNewGroup(SIRType.ID_SUSCEPTIBLE.ordinal(), Integer.MAX_VALUE/2);
				getGroupsById().put(SIRType.ID_SUSCEPTIBLE.ordinal(), g);
			}
			return SIRType.ID_SUSCEPTIBLE.ordinal();
		}
	}


	@Override
	public void registerGroupSizeDeterminator(int sourceId, GroupSizeDeterminator gsD) {
		sourceNextGroups.put(sourceId, new LinkedList<>());
	}

	@Override
	public int nextGroupForSource(int sourceId) {
		return Integer.MAX_VALUE/2;
	}

	@Override
	public SIRGroup getGroup(final Pedestrian pedestrian) {
		SIRGroup group = groupsById.get(pedestrian.getGroupIds().getFirst());
		assert group != null : "No group found for pedestrian";
		return group;
	}

	@Override
	protected void registerMember(final Pedestrian ped, final SIRGroup group) {
		groupsById.putIfAbsent(ped.getGroupIds().getFirst(), group);
	}

	@Override
	public Map<Integer, SIRGroup> getGroupsById() {
		return groupsById;
	}

	@Override
	protected SIRGroup getNewGroup(final int size) {
		return getNewGroup(getFreeGroupId(), size);
	}

	@Override
	protected SIRGroup getNewGroup(final int id, final int size) {
		if(groupsById.containsKey(id))
		{
			return groupsById.get(id);
		}
		else
		{
			return new SIRGroup(id, this);
		}
	}

	private void initializeGroupsOfInitialPedestrians() {
		// get all pedestrians already in topography
		DynamicElementContainer<Pedestrian> c = topography.getPedestrianDynamicElements();

		if (c.getElements().size() > 0) {
			// Here you can fill in code to assign pedestrians in the scenario at the beginning (i.e., not created by a source)
            //  to INFECTED or SUSCEPTIBLE groups. This is not required in the exercise though.
		}
	}

	protected void assignToGroup(Pedestrian ped, int groupId) {
		SIRGroup currentGroup = getNewGroup(groupId, Integer.MAX_VALUE/2);
		currentGroup.addMember(ped);
		ped.getGroupIds().clear();
		ped.getGroupSizes().clear();
		ped.addGroupId(currentGroup.getID(), currentGroup.getSize());
		registerMember(ped, currentGroup);
	}

	protected void assignToGroup(Pedestrian ped) {
		int groupId = getFreeGroupId();
		assignToGroup(ped, groupId);
	}


	/* DynamicElement Listeners */

	@Override
	public void elementAdded(Pedestrian pedestrian) {
		assignToGroup(pedestrian);
	}

	@Override
	public void elementRemoved(Pedestrian pedestrian) {
		Group group = groupsById.get(pedestrian.getGroupIds().getFirst());
		if (group.removeMember(pedestrian)) { // if true pedestrian was last member.
			groupsById.remove(group.getID());
		}
	}

	/* Model Interface */

	@Override
	public void preLoop(final double simTimeInSec) {
		initializeGroupsOfInitialPedestrians();
		topography.addElementAddedListener(Pedestrian.class, this);
		topography.addElementRemovedListener(Pedestrian.class, this);
	}

	@Override
	public void postLoop(final double simTimeInSec) {
	}

	@Override
	public void update(final double simTimeInSec) {
		/* Performs one step of the simulation by checking all neighbors of each pedestrian, and
		 updates pedestrian states to infected or recovered depending on the maximum infection and recovery rate.

        Arguments:
        ----------
        simTimeInSec : double
            Gives the total time passed up to the simulation step.

        Returns:
        --------
        Nothing: Only does updates on the simulation backend states to be visualized on the grid.
        Doesn't need to return anything.

        */

		// Get the time step of the simulation by calculating the difference between two steps
		double deltaTimeInSec = simTimeInSec - lastSimTime;
		lastSimTime = simTimeInSec; // Assign last time step

		DynamicElementContainer<Pedestrian> c = topography.getPedestrianDynamicElements(); // Get all pedestrians
		if (c.getElements().isEmpty()) return; // No updates if there are 0 pedestrians

		Rectangle2D bounds2D = topography.getBounds(); // Get bounds of topography in 2D
		// Create VRectangle bounds for grid to use in the constructor
		VRectangle bounds = new VRectangle(bounds2D.getX(), bounds2D.getY(), bounds2D.getWidth(), bounds2D.getHeight());
		// Heuristic for side length, doesn't matter since grid is there for checking neighbors
		double cellSize = attributesSIRG.getInfectionMaxDistance();

		LinkedCellsGrid<Pedestrian> grid = new LinkedCellsGrid<>(bounds, cellSize); // Create linked cell grid
		for (Pedestrian p : c.getElements()) grid.addObject(p); // Add all pedestrians to the grid

		double infectionRatePerSecond = attributesSIRG.getInfectionRate(); // Get infection rate
		// Calculate infection probability with the given formula in the task
		double infectionProbThisStep = 1.0 - Math.pow(1.0 - infectionRatePerSecond, deltaTimeInSec);

		double recoveryRatePerSecond = attributesSIRG.getRecoveryRate(); // Get recovery rate
		// Calculate recovery probability with the given formula in the task
		double recoveryProbThisStep = 1.0 - Math.pow(1.0 - recoveryRatePerSecond, deltaTimeInSec);

		for(Pedestrian p : c.getElements()) { // For each pedestrian

			// If the pedestrian is already infected, there is a chance for recovery
			if (getGroup(p).getID() == SIRType.ID_INFECTED.ordinal()) { // Check if pedestrian is infected
				if (random.nextDouble() < recoveryProbThisStep) { // If recovered
					elementRemoved(p); // Remove pedestrian
					assignToGroup(p, SIRType.ID_REMOVED.ordinal()); // Assign to recovered group
					continue; // Continue if pedestrian recovered
				}
			}

			// If the pedestrian isn't susceptible (already infected or recovered), no need to check
			if (getGroup(p).getID() != SIRType.ID_SUSCEPTIBLE.ordinal()) continue;

			// Get all pedestrians in the given area, which in this case is the maximum infection distance
			List<Pedestrian> neighbors = grid.getObjects(p.getPosition(), cellSize);

			for(Pedestrian p_neighbor : neighbors) { // For each neighbor
				if(p == p_neighbor) continue; // Continue if neighbor is the same

				if (getGroup(p_neighbor).getID() == SIRType.ID_INFECTED.ordinal()) { // If neighbor isn't already infected
					if (random.nextDouble() < infectionProbThisStep) { // Check for infection possibility given probability formula
						elementRemoved(p); // Remove non-infected pedestrian
						assignToGroup(p, SIRType.ID_INFECTED.ordinal()); // Create infected pedestrian in its place
						break; // Break to not check remaining neighbors
					}
				}
			}
		}
	}
}