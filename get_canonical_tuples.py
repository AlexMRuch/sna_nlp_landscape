"""
Script: get_canonical_tuples.py
Purpose: Process a pandas DataFrame into knowledge graph model inputs
Outputs:
    1. entities.tsv: encoding-entity mapping
    2. relations.tsv: encoding-relation mapping
    3. canonical_tuples_k.tsv: all (source, relation, destination) canonical tuples
        Note: source, relation, and destination values are encoded elements
        Note: canonical_tuples_k.tsv excludes tuples with poorly connected destination entities (k)
    4. train.tsv: encoded canonical tuples training split
    5. valid.tsv: encoded canonical tuples validation split
    6. test.tsv: encoded canonical tuples testing split
sysargs:
    1. output path (set to folder to save results)
    2. use swifter (vs. standard pandas)
    3. training split proportion (set to float)
    4. validation split proportion (set to float)
    5. testing split proportion (set to float)
    6. k-core cut for destination nodes (set to int)
    Note: train+valid+test floats must add to 1.0
Runtime:
    python get_canonical_tuples.py /media/seagate0/reddit/models/kg/ 1 0.9 0.05 0.05 2
"""

# Import dependencies
import sys
import math
import numpy as np
import pandas as pd
import swifter
import csv
import datetime
from sklearn.model_selection import train_test_split


# Run script from terminal with input and output parameters
if __name__ == "__main__":

    # Setup script
    time_start_start = datetime.datetime.now()
    path_output = str(sys.argv[1])
    use_swifter = int(sys.argv[2])
    train_pct   = float(sys.argv[3])
    valid_pct   = float(sys.argv[4])
    test_pct    = float(sys.argv[5])
    k           = int(sys.argv[6])
    assert round(sum([train_pct+valid_pct+test_pct]), 6) == 1.0, "train_pct+valid_pct+test_pct != 1.0"
    if valid_pct < 0.05:
        print("Warning: valid_pct is < 0.05, which may result in poor training")
    if test_pct < 0.05:
        print("Warning: test_pct is < 0.05, which may result in poor testing")
    if k < 5:
        print("Warning: k < 5, which may cause errors splitting training, validation, testing sets")


    # Load data
    print("Loading graph data...")
    graph_df = pd.read_csv("/media/seagate0/reddit/samples/sample_main_preprocessed.csv")
    print("  Graph dataframe shape:", graph_df.shape)
    authors_list = np.array(list(set(
        graph_df["submission_author"].unique().tolist() +
        graph_df["comment_author"].unique().tolist()
    ))).tolist()
    print("  Number of unique authors in graph dataframe:", len(authors_list))
    submissions_list = graph_df["submission_id"].unique().tolist()
    print("  Number of unique submissions in graph dataframe:", len(submissions_list))
    subreddits_list = graph_df["subreddit"].unique().tolist()
    if np.nan in subreddits_list:  # hardcode the r/NaN subreddit (reddit.com/r/NaN/) as string
        print("    Warning: r/NaN subreddit found, hardcoding as 'NaN'...")
        nan_idx = subreddits_list.index(np.nan)
        subreddits_list[nan_idx] = "NaN"
        for idx,subreddit_name in enumerate(graph_df["subreddit"]):
            try:
                if math.isnan(subreddit_name) == True:
                    graph_df.loc[idx, "subreddit"] = "NaN"
            except:
                pass
    print("  Number of unique subreddits in graph dataframe:", len(subreddits_list))


    # Create encodings
    ## Create entity set and entity-encoding map
    print("\nCreating entity set and entity-encoding map")
    entity_set = set(authors_list + submissions_list + subreddits_list)
    assert np.nan not in entity_set, \
        "Processing error: entity set has nan value(s)"
    print(f"  Encoding {len(entity_set)} entities...")
    entity_encoding_dict = {entity:encoding for (encoding,entity) in enumerate(entity_set)}
    print(f"  Encoded {len(entity_encoding_dict)} entities")
    assert len(entity_encoding_dict) == len(entity_set), \
        "Processing error: lengths of entity set and entity encodings unequal"

    ## Create bidirectional relation set and relation-encoding map
    print("\nCreating bidirectional relation set and relation-encoding map...")
    relation_set = set([
        "/author_writes_submission",  # authors and submissions
        "/submission_written_by_author",
        "/author_comments_on_submission",  # authors - comments - submissions
        "/submission_comment_from_author",
        "/submission_to_subreddit",  # submissions and subreddits
        "/subreddit_receives_submission"
    ])
    print(f"  Encoding {len(relation_set)} relations...")
    relation_encoding_dict = {relation:encoding for (encoding,relation) in enumerate(relation_set)}
    print(f"  Encoded {len(relation_encoding_dict)} relations")
    assert len(relation_encoding_dict) == len(relation_set), \
        "Processing error: lengths of relation set and relation encodings unequal"

    # Encode entities
    print("\nEncoding entities...")
    entity_col_list = [
        "submission_author",
        "comment_author",
        "submission_id",
        "subreddit"
    ]
    for entity_col in entity_col_list:
        print(f"  Encoding {entity_col} from ...")
        graph_df[entity_col] = graph_df[entity_col].apply(lambda x: entity_encoding_dict[x])
        print("  to ...\n", graph_df[entity_col].head())


    # Create canonical tuples
    print("\nCreating forward-running canonical tuples dataframe...")
    time_start = datetime.datetime.now()
    canonical_tuples = pd.DataFrame(columns=['src', 'rel', 'dst'])
    relation_cols_dict_fwd = {
        "/author_writes_submission": ["submission_author", "submission_id"],  # authors and submissions
        "/author_comments_on_submission": ["comment_author", "submission_id"],  # authors - comments - submissions
        "/submission_to_subreddit": ["submission_id", "subreddit"]  # submissions and subreddits
    }
    relation_cols_dict_rev = {
        "/author_writes_submission": "/submission_written_by_author",  # submissions and authors
        "/author_comments_on_submission": "/submission_comment_from_author",  # submissions - comments - authors
        "/submission_to_subreddit": "/subreddit_receives_submission"  # subreddits and submissions
    }
    for (relation, relation_cols) in relation_cols_dict_fwd.items():
        print(f"  Extracting forward-running canonical tuples for {relation} ...")
        relation_encoding = relation_encoding_dict[relation]
        tmp_df = graph_df.loc[:,relation_cols].rename(
            columns={relation_cols[0]: "src", relation_cols[1]: "dst"}
        )
        tmp_df["rel"] = relation_encoding
        tmp_df = tmp_df[["src", "rel", "dst"]]
        canonical_tuples = pd.concat([canonical_tuples, tmp_df]).reset_index(drop=True)
        print(f"  Extracted {len(canonical_tuples)} forward-running canonical tuples:\n", canonical_tuples.tail())
    print("Created forward-running canonical tuples dataframe:", canonical_tuples.shape)
    assert canonical_tuples.shape[1] == 3, \
        "Processing error: forward-running canonical tuples width ≠ 3"


    # Perform k-core decomposition on forward-running canonical tuples
    canonical_tuples_len_before = len(canonical_tuples)
    print(f"\nPerform k-core decomposition on forward-running canonical tuples (k = {k}) ...")
    if use_swifter == False:
        canonical_tuples = np.array(canonical_tuples)
    print("  Extracting unique encoded source entity values...")
    if use_swifter == True:
        uniqueValues_src, occurCount_src = np.unique(canonical_tuples["src"].values, return_counts=True)
    else:
        uniqueValues_src, occurCount_src = np.unique(canonical_tuples[:,0], return_counts=True)
    print("  Number of unique source entities before dropping poorly connected entities:     ", len(uniqueValues_src))
    ## Get values of poorly connected source entities (entities with < k edges)
    print("  Finding short source values...")
    short_uniqueValues_src = uniqueValues_src[np.where(occurCount_src < k, True, False)]
    ## Get indices of poorly connected source enttiies (entities with < k edges)
    print(f"  Found {len(short_uniqueValues_src)} short source values")
    print("  Finding short source values' indices...")
    if use_swifter == True:
        short_indices_src = canonical_tuples["src"].swifter.allow_dask_on_strings().apply(
            lambda x: x in short_uniqueValues_src
        ).values
    else:
        short_indices_src = np.in1d(canonical_tuples[:,0], short_uniqueValues_src)
    print(f"  Found {len(short_indices_src)} short source value indices")
    ## Count unique destination values for splitting
    print("  Extracting unique encoded destination entity values...")
    if use_swifter == True:
        uniqueValues_dst, occurCount_dst = np.unique(canonical_tuples["dst"].values, return_counts=True)
    else:
        uniqueValues_dst, occurCount_dst = np.unique(canonical_tuples[:,2], return_counts=True)
    print("  Number of unique destination entities before dropping poorly connected entities:", len(uniqueValues_dst))
    ## Get values of poorly connected destination entities (entities with < k edges)
    print("  Finding short destination values...")
    short_uniqueValues_dst  = uniqueValues_dst[np.where(occurCount_dst < k, True, False)]
    ## Get indices of poorly connected destination enttiies (entities with < k edges)
    print(f"  Found {len(short_uniqueValues_dst)} short destination values")
    print("  Finding short destination values' indices...")
    if use_swifter == True:
        short_indices_dst  = canonical_tuples["dst"].swifter.allow_dask_on_strings().apply(
            lambda x: x in short_uniqueValues_dst
        ).values
    else:
        short_indices_dst  = np.in1d(canonical_tuples[:,2], short_uniqueValues_dst)
    print(f"  Found {len(short_indices_dst)} short destination value indices")
    ## Get union of Boolean array of src/dst short indices by adding vectors
    print("  Combining index markers for short source and destination entities...")
    short_indices_union = short_indices_src + short_indices_dst
    ## Keep rows with strongly connected source and destination entities
    print("  Dropping rows with short source or destination entities...")
    if use_swifter == True:
        canonical_tuples = canonical_tuples.values[~short_indices_union]
    else:
        canonical_tuples = canonical_tuples[~short_indices_union]
    ## Get length of canonical tuples and source/destination entities after dropping poorly connected rows
    print("Getting post-drop statistics...")
    canonical_tuples_len_after = len(canonical_tuples)
    print("  Number of forward-running canonical tuples after dropping poorly connected entities:            ", canonical_tuples_len_after)
    print(f"    {canonical_tuples_len_before - canonical_tuples_len_after} rows removed")
    uniqueValues_src_after     = np.unique(canonical_tuples[:,0])
    uniqueValues_src_len_after = len(uniqueValues_src_after)
    print("  Number of unique source entities after dropping poorly connected entities:      ", uniqueValues_src_len_after)
    uniqueValues_dst_len_after = len(np.unique(canonical_tuples[:,2]))
    print("  Number of unique destination entities after dropping poorly connected entities: ", uniqueValues_dst_len_after)
    uniqueValues_rel_after     = np.unique(canonical_tuples[:,1])
    uniqueValues_rel_len_after = len(uniqueValues_rel_after)
    print("  Number of unique relations after dropping poorly connected entities:            ", uniqueValues_rel_len_after)


    # Define reverse encoding dicts (otherwise embedding matrix will be too large)
    print("\nCreate reverse encoding dicts for entities and relations...")
    encoding_entity_dict  = {v:k for (k,v) in entity_encoding_dict.items()}
    encoding_relation_dict = {v:k for (k,v) in relation_encoding_dict.items()}
    print("  Created reverse encoding dicts for entities and relations")

    # Apply reverse encoding dicts to convert encodings to raw entity/relation names
    print("\nReverse encode entities and relations to reset encodings for KG modeling...")
    canonical_tuples = pd.DataFrame(canonical_tuples, columns=["src","rel","dst"])
    canonical_tuples["src"] = canonical_tuples["src"].apply(lambda x: encoding_entity_dict[x])
    canonical_tuples["rel"] = canonical_tuples["rel"].apply(lambda x: encoding_relation_dict[x])
    canonical_tuples["dst"] = canonical_tuples["dst"].apply(lambda x: encoding_entity_dict[x])
    print("  Reversed encode entities and relations to reset encodings for KG modeling")

    # Make entities map for encoding
    ## Create new encoding-entity map
    print("\nCreating new encoding-entity map (reset indices from 0:len(entities))...")
    uniqueValues_ent_after_decoded = set(
        canonical_tuples["src"].unique().tolist() +
        canonical_tuples["dst"].unique().tolist()
    )
    entities = []  # structure as [(ent_id, ent_name)] for entities.tsv output
    for e in uniqueValues_ent_after_decoded:
        entities.append([len(entities), str(e)])
    print("  Extracted {} unique entities".format(len(entities)))
    print("  Examples of encoded entities:\n", entities[:30], "...")
    ## Create tmp obj for entity name - entity id encodings
    print("  Creating entity-encoding map...")
    entity_encoding_dict = {entity:encoding for (encoding,entity) in entities}
    print("  Created entity-encoding map for {} entities".format(len(entity_encoding_dict)))

    # Make relations map for encoding
    ## Create new encoding-relation map
    print("\nCreating encoding-relation map (reset indices from 0:len(relations))...")
    relations = []  # structure as [(rel_id, rel_name)] for relations.tsv output
    for r in relation_set:
        relations.append([len(relations), str(r)])
    print("  Extracted {} unique relations".format(len(relations)))
    print("  Examples of encoded relations:\n", relations)
    ## Create tmp obj for relation name - relation id encodings
    print("  Creating relation-encoding map...")
    relation_encoding_dict = {relation:encoding for (encoding,relation) in relations}
    print("  Created relation-encoding map for {} relations".format(len(relation_encoding_dict)))

    ## Encode entities, relations, and canonical tuples
    print("\nEncode entities, relations, and canonical tuples...")
    ## Encode canonical tuples
    print("  Encoding forward-running canonical tuples...")
    canonical_tuples["src"] = canonical_tuples["src"].apply(lambda x: entity_encoding_dict[x])
    canonical_tuples["rel"] = canonical_tuples["rel"].apply(lambda x: relation_encoding_dict[x])
    canonical_tuples["dst"] = canonical_tuples["dst"].apply(lambda x: entity_encoding_dict[x])
    print(f"  Encoded {len(canonical_tuples)} canonical tuples")
    print("  Examples of canonical tuples:\n", canonical_tuples[:30])

    ## Add reverse-running canonical tuples
    print("\nCreating reverse-running canonical tuples dataframe...")
    canonical_tuples_len = len(canonical_tuples)
    canonical_tuples_rev = canonical_tuples[["dst", "rel", "src"]]
    for (fwd_relation, rev_relation) in relation_cols_dict_rev.items():
        print(f"  Recoding {fwd_relation} forward-relations as {rev_relation} for reverse-running canonical tuples ...")
        fwd_relation_encoding = relation_encoding_dict[fwd_relation]
        rev_relation_encoding = relation_encoding_dict[rev_relation]
        canonical_tuples_rev["rel"].replace({fwd_relation_encoding: rev_relation_encoding}, inplace=True)
    print("  Concatenating reverse-running canonical tuples to forward-running canonical tuples ...")
    canonical_tuples = pd.concat([canonical_tuples, canonical_tuples_rev]).reset_index(drop=True)
    print(f"  Extracted {len(canonical_tuples)} reverse-running canonical tuples:\n", canonical_tuples.tail())
    print("Created reverse-running canonical tuples dataframe:", canonical_tuples.shape)
    assert canonical_tuples.shape[0] == canonical_tuples_len*2, \
        "Processing error: fwd+rev canonical tuples length ≠ 2x fwd length"
    assert canonical_tuples.shape[1] == 3, \
        "Processing error: canonical tuples width ≠ 3"

    ## Update new encoding-relation map
    print("\nCreating encoding-relation map, post-concatenating (reset indices from 0:len(relations))...")
    uniqueValues_rel_after_decoded = canonical_tuples["rel"].unique()
    relations = []  # structure as [(rel_id, rel_name)] for relations.tsv output
    for r in uniqueValues_rel_after_decoded:
        relations.append([len(relations), str(r)])
    print("  Extracted {} unique relations, post-concatenating".format(len(relations)))
    print("  Examples of encoded relations:\n", relations)
    ## Create tmp obj for relation name - relation id encodings
    print("  Creating post-concatenating relation-encoding map...")
    relation_encoding_dict = {relation:encoding for (encoding,relation) in relations}
    print("  Created post-concatenating relation-encoding map for {} relations".format(len(relation_encoding_dict)))


    # Save encoded entities, relations, and canonical tuples
    print("\nSave encoded entities, relations, and canonical tuples")
    print("\n  Saving encoding-entity map as entities.tsv (for KG models) ...")
    canonical_tuples = np.array(canonical_tuples)
    with open(path_output+'entities.tsv', 'w') as f:
        write_counter = 0
        entities_len = len(entities)
        write_reporter = entities_len//20
        tsv_writer = csv.writer(f, delimiter='\t')
        for i in entities:
            try:
                tsv_writer.writerow(i)
            except:
                try:
                    i = (i[0], i[1].encode('utf-8', 'surrogatepass').decode('ISO-8859-1'))
                    tsv_writer.writerow(i)
                except Exception as e:
                    print(e)
                    print(i)
            write_counter += 1
            if write_counter==1 or write_counter%write_reporter==0:
                print("  Saved {}/{} entities".format(
                    write_counter,
                    entities_len
                ))
    print("  Saved", len(entities), "entities to", path_output+'entities')
    print("\n  Saving encoding-relation map as relations.tsv (for KG models) ...")
    with open(path_output+'relations.tsv', 'w') as f:
        write_counter = 0
        relations_len = len(relations)
        write_reporter = relations_len
        tsv_writer = csv.writer(f, delimiter='\t')
        for i in relations:
            try:
                tsv_writer.writerow(i)
            except:
                try:
                    i = (i[0], i[1].encode('utf-8', 'surrogatepass').decode('ISO-8859-1'))
                    tsv_writer.writerow(i)
                except Exception as e:
                    print(e)
                    print(i)
            write_counter += 1
            if write_counter==1 or write_counter%write_reporter==0:
                print("  Saved {}/{} relations".format(
                    write_counter,
                    relations_len
                ))
    print("  Saved", len(relations), "relations to", path_output+'relations')
    print("\n  Saving canonical tuples...")
    with open(path_output+'canonical_tuples_{}.tsv'.format(k), 'w') as f:
        write_counter = 0
        canonical_tuples_len = len(canonical_tuples)
        write_reporter = canonical_tuples_len//20
        tsv_writer = csv.writer(f, delimiter='\t')
        for i in canonical_tuples:
            try:
                tsv_writer.writerow(i)
            except:
                try:
                    i = tuple(s.encode('utf-8', 'surrogatepass').decode('ISO-8859-1') for s in i)
                    tsv_writer.writerow(i)
                except Exception as e:
                    print(e)
                    print(i)
            write_counter += 1
            if write_counter==1 or write_counter%write_reporter==0:
                print("  Saved {}/{} canonical tuples".format(
                    write_counter,
                    canonical_tuples_len
                ))
    print("  Saved", len(canonical_tuples), "canonical tuples to", path_output+'canonical_tuples.tsv')

    # Print runtime
    time_end = datetime.datetime.now()
    time_run = time_end-time_start
    print("\nFinished filtering and encoding pipeline:", time_end)
    time_run_sec = time_run.total_seconds()
    print(f"  Total runtime: {time_run_sec} seconds (== {time_run_sec/60} mins)")


    # Split canonical tuples into train/valid/test splits
    time_start = datetime.datetime.now()
    print("\nBegin training-validation-testing split pipeline:", time_start)
    # Get train-valid vs test splits
    print("Extracting test set...")
    X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(
        canonical_tuples[:,:2],
        canonical_tuples[:,2],
        stratify=canonical_tuples[:,2],
        test_size=test_pct,
        random_state=407
    )
    # Get train vs valid splits
    print("\nExtracting training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_and_val,
        y_train_and_val,
        stratify=y_train_and_val,
        test_size=valid_pct,
        random_state=407
    )
    # Stack arrays to recreate tuples
    train = np.column_stack((X_train,y_train))
    valid = np.column_stack((X_val,y_val))
    test  = np.column_stack((X_test,y_test))
    print("Training set shape:     ", train.shape)
    print("  Training set head:  \n", train[:5])
    print("Validation set shape:   ", valid.shape)
    print("  Validation set head:\n", valid[:5])
    print("Test set shape:         ", test.shape)
    print("  Testing set head:   \n", test[:5])

    # Save training set
    print("\nSaving training set...")
    with open(path_output+'train.tsv', 'w') as f:
        write_counter = 0
        train_len = len(train)
        write_reporter = train_len//20
        tsv_writer = csv.writer(f, delimiter='\t')
        for i in train:
            try:
                tsv_writer.writerow(i)
            except:
                try:
                    i = tuple(s.encode('utf-8', 'surrogatepass').decode('ISO-8859-1') for s in i)
                    tsv_writer.writerow(i)
                except Exception as e:
                    print(e)
                    print(i)
            write_counter += 1
            if write_counter==1 or write_counter%write_reporter==0:
                print("  Saved {}/{} training canonical tuples".format(
                    write_counter,
                    train_len
                ))
    print(f"  Wrote {len(train)} tuples to train.tsv")
    # Save validation set
    print("\nSaving validation set...")
    with open(path_output+'valid.tsv', 'w') as f:
        write_counter = 0
        valid_len = len(valid)
        write_reporter = valid_len//20
        tsv_writer = csv.writer(f, delimiter='\t')
        for i in valid:
            try:
                tsv_writer.writerow(i)
            except:
                try:
                    i = tuple(s.encode('utf-8', 'surrogatepass').decode('ISO-8859-1') for s in i)
                    tsv_writer.writerow(i)
                except Exception as e:
                    print(e)
                    print(i)
            write_counter += 1
            if write_counter==1 or write_counter%write_reporter==0:
                print("  Saved {}/{} validation canonical tuples".format(
                    write_counter,
                    valid_len
                ))
    print(f"  Wrote {len(valid)} tuples to valid.tsv")
    # Save testing set
    print("\nSaving testing set...")
    with open(path_output+'test.tsv', 'w') as f:
        write_counter = 0
        test_len = len(test)
        write_reporter = test_len//20
        tsv_writer = csv.writer(f, delimiter='\t')
        for i in test:
            try:
                tsv_writer.writerow(i)
            except:
                try:
                    i = tuple(s.encode('utf-8', 'surrogatepass').decode('ISO-8859-1') for s in i)
                    tsv_writer.writerow(i)
                except Exception as e:
                    print(e)
                    print(i)
            write_counter += 1
            if write_counter==1 or write_counter%write_reporter==0:
                print("  Saved {}/{} testing canonical tuples".format(
                    write_counter,
                    test_len
                ))
    print(f"  Wrote {len(test)} tuples to test.tsv")

    # Print runtime
    time_end = datetime.datetime.now()
    time_run = time_end-time_start
    print("\nFinished training-validation-testing split pipeline:", time_end)
    time_run_sec = time_run.total_seconds()
    print(f"  Total runtime: {time_run_sec} seconds (== {time_run_sec/60} mins)")


    # Report total runtime
    time_end_end = datetime.datetime.now()
    time_run_run = time_end_end-time_start_start
    print("\n\nFinished script:", time_end_end)
    time_run_run_sec = time_run_run.total_seconds()
    print(f"  Total script runtime: {time_run_run_sec} seconds (== {time_run_run_sec/60} mins)")
