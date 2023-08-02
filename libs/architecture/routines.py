import torch
import torch.nn as nn


def hmdb(unified_transformer, videos, targets=None, mode='train', return_str_preds=False, num_steps=1):
    '''
    Performs the HMDB task.

    Args:
        unified_transformer (UnifiedTransformer): The unified transformer.
        videos (torch.Tensor): The input videos.
        targets (torch.Tensor, optional): The target labels. Defaults to None.
        mode (str, optional): The mode. Defaults to 'train'.
        return_string_predictions (bool, optional): Whether to return predictions in string format. Defaults to False.
        num_steps (int, optional): The number of decoding steps. Defaults to 1.

    Returns:
        torch.Tensor: The predictions.
        torch.Tensor: The loss.
        torch.Tensor: The accuracy.
    '''
    batch_size = videos.shape[0]

    # Reset the unified transformer state
    unified_transformer.reset(batch_size)

    # Encode video files
    unified_transformer.encode_videos(videos, domain='IMAGE')
    predictions, loss, acc = None, None, None # Set loss and accuracy to None if not provided
    
    if mode in ['train', 'val']:
        # Decode from targets
        predictions = unified_transformer.decode_from_targets('HMDB', targets=targets)
        # Calculate loss if targets is provided
        loss, acc = calc_nll_loss_and_acc(predictions, targets)
    elif mode == 'predict':
        # Decode using greedy decoding
        predictions = unified_transformer.decode_greedy('HMDB', num_steps=num_steps)

    if return_str_preds and predictions is not None:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)

    return predictions, loss, acc



def vqa(unified_transformer, images, questions, targets=None, mode='train', return_str_preds=False, num_steps=1):
    '''
    Performs the VQA task.

    Args:
        unified_transformer (UnifiedTransformer): The unified transformer.
        images (torch.Tensor): The input images.
        questions (list): The input questions.
        targets (torch.Tensor, optional): The target labels. Defaults to None.
        mode (str, optional): The mode. Defaults to 'train'.
        return_string_predictions (bool, optional): Whether to return predictions in string format. Defaults to False.
        num_steps (int, optional): The number of decoding steps. Defaults to 1.

    Returns:
        torch.Tensor: The predictions.
        torch.Tensor: The loss.
        torch.Tensor: The accuracy.
    '''
    # Reset the unified transformer state
    batch_size = images.shape[0]
    unified_transformer.reset(batch_size)

    # Encode and store images
    unified_transformer.encode_images(images, domain='IMAGE')

    # Encode and store questions
    unified_transformer.encode_englishtexts(questions)
    predictions, loss, acc = None, None, None

    if mode in ['train', 'val']:
        # Decode from targets
        predictions = unified_transformer.decode_from_targets('VQA', targets=targets)
    elif mode == 'predict':
        # Decode using greedy decoding
        predictions = unified_transformer.decode_greedy('VQA', num_steps=num_steps)

    # Calculate loss if targets is provided
    if targets is not None:
        loss, acc = calc_nll_loss_and_acc(predictions, targets)
    else:
        loss, acc = None, None

    if return_str_preds and predictions is not None:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)

    return predictions, loss, acc


def image_caption(unified_transformer, images, targets=None, mode='train', return_str_preds=False, num_steps=100):
    '''
    Performs the image captioning task.

    Args:
        unified_transformer (UnifiedTransformer): The unified transformer.
        images (torch.Tensor): The input images.
        targets (list, optional): The target captions. Defaults to None.
        mode (str, optional): The mode. Defaults to 'train'.
        return_string_predictions (bool, optional): Whether to return predictions in string format. Defaults to False.
        num_steps (int, optional): The number of decoding steps. Defaults to 100.

    Returns:
        torch.Tensor: The predictions.
        torch.Tensor: The loss.
        torch.Tensor: The accuracy.
    '''
    # Reset the unified transformer state
    batch_size = images.shape[0]
    unified_transformer.reset(batch_size)

    # Encode and store images
    unified_transformer.encode_images(images, domain='IMAGE')

    target_pad_mask = None
    if targets is not None:
        # Tokenize target captions and calculate pad mask
        targets, target_pad_mask = unified_transformer.english_language_perph.tokenize_sentences(targets)

    predictions, loss, acc = None, None, None

    if mode in ['train', 'val']:
        # Decode from targets
        predictions = unified_transformer.decode_from_targets('IMAGE_CAPTION', targets=targets, target_pad_mask=target_pad_mask)
    elif mode == 'predict':
        # Decode using greedy decoding
        predictions = unified_transformer.decode_greedy('IMAGE_CAPTION', num_steps=num_steps)

    # Calculate loss if targets is provided
    if targets is not None:
        pad_token = unified_transformer.english_language_perph.id_PAD
        loss, acc = calc_nll_loss_and_acc(predictions, targets, pad_id=pad_token, target_pad_mask=target_pad_mask)
    else:
        loss, acc = None, None

    if return_str_preds and predictions is not None:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)
        predictions = unified_transformer.english_language_perph.decode_tokens(predictions)

    return predictions, loss, acc


def penn(unified_transformer, texts, pad_id=None, targets=None, target_pad_mask=None, mode='train', return_str_preds=False, num_steps=100):
    '''
    Performs the Penn Treebank task.

    Args:
        unified_transformer (UnifiedTransformer): The unified transformer.
        texts (list): The input texts.
        pad_id (int, optional): The padding id. Defaults to None.
        targets (list, optional): The target texts. Defaults to None.
        target_pad_mask (torch.Tensor, optional): The target padding mask. Defaults to None.
        mode (str, optional): The mode. Defaults to 'train'.
        return_string_predictions (bool, optional): Whether to return predictions in string format. Defaults to False.
        num_steps (int, optional): The number of decoding steps. Defaults to 100.

    Returns:
        torch.Tensor: The predictions.
        torch.Tensor: The loss.
        torch.Tensor: The accuracy.
    '''
    # Reset the unified transformer state
    batch_size = len(texts)
    unified_transformer.reset(batch_size)

    # Encode and store texts
    unified_transformer.encode_englishtexts(texts, domain='ENGLISH')

    predictions, loss, acc = None, None, None

    if mode in ['train', 'val']:
        # Decode from targets
        predictions = unified_transformer.decode_from_targets('PENN', targets=targets, target_pad_mask=target_pad_mask)
    elif mode == 'predict':
        # Decode using greedy decoding
        predictions = unified_transformer.decode_greedy('PENN', num_steps=num_steps)

    # Calculate loss if targets is provided
    if targets is not None:
        loss, acc = calc_nll_loss_and_acc(predictions, targets, pad_id=pad_id, target_pad_mask=target_pad_mask)
    else:
        loss, acc = None, None

    if return_str_preds and predictions is not None:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)
        predictions = unified_transformer.english_language_perph.decode_tokens(predictions)

    return predictions, loss, acc
 

def calc_nll_loss_and_acc(predictions, targets, pad_id=None, target_pad_mask=None):
    '''
    Calculates the negative log likelihood loss and accuracy.

    Args:
        predictions (torch.Tensor): The predictions.
        targets (torch.Tensor): The target values.
        pad_id (int, optional): The padding id. Defaults to None.
        target_pad_mask (torch.Tensor, optional): The target padding mask. Defaults to None.

    Returns:
        torch.Tensor: The loss.
        torch.Tensor: The accuracy.
    '''
    # Calculate loss
    predictions_flat = torch.reshape(predictions, [-1, predictions.shape[2]])
    if pad_id is not None:
        loss_fn = nn.NLLLoss(ignore_index=pad_id)
    else:
        loss_fn = nn.NLLLoss()
    targets_flat = torch.reshape(targets, [-1])
    loss = loss_fn(predictions_flat, targets_flat)

    # Calculate accuracy
    predictions_argmax = predictions.argmax(-1)
    predictions_argmax_flat = torch.reshape(predictions_argmax, [-1])
    if target_pad_mask is not None:
        target_pad_mask_flat = torch.reshape(target_pad_mask, [-1])
        predictions_argmax_flat = predictions_argmax_flat + (target_pad_mask_flat * 1000000).to(dtype=torch.long)
        accuracy = (torch.sum(targets_flat == predictions_argmax_flat).sum().cpu().numpy() / (targets_flat.shape[0] - target_pad_mask_flat.sum().cpu().numpy())) * 100
    else:
        accuracy = (torch.sum(targets_flat == predictions_argmax_flat).sum().cpu().numpy() / (targets_flat.shape[0])) * 100

    return loss, accuracy