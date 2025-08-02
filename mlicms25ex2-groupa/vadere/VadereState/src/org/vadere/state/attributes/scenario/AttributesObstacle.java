package org.vadere.state.attributes.scenario;

import org.vadere.state.scenario.ScenarioElement;
import org.vadere.util.geometry.shapes.VShape;
import java.util.Objects;

/**
 * An Obstacle is a {@link ScenarioElement} which blocks agents and cannot be passed by them.
 * <warning-box>If used with a OptimalStepsModel a width / height of &lt; 0.1 may lead to problems of agents ignoring it.</warning-box>
 */
public class AttributesObstacle extends AttributesVisualElement{

	public AttributesObstacle() {}

	public AttributesObstacle(int id) {
		this.id = id;
	}

	public AttributesObstacle(int id, VShape shape) {
		this(id);
		this.shape = shape;
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (o == null || getClass() != o.getClass()) return false;

		AttributesObstacle that = (AttributesObstacle) o;

		if (id != that.id) return false;
		return Objects.equals(shape, that.shape);
	}

	@Override
	public int hashCode() {
		int result = shape != null ? shape.hashCode() : 0;
		result = 31 * result + id;
		return result;
	}
}