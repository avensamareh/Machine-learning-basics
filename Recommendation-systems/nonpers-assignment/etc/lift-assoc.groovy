import org.lenskit.api.ItemBasedItemRecommender
import org.lenskit.mooc.nonpers.assoc.LiftAssociationModelProvider
import org.lenskit.mooc.nonpers.assoc.AssociationItemBasedItemRecommender
import org.lenskit.mooc.nonpers.assoc.AssociationModel

bind ItemBasedItemRecommender to AssociationItemBasedItemRecommender
bind AssociationModel toProvider LiftAssociationModelProvider
