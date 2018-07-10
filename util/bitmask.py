# -*- coding: utf-8 -*-
# Licensed under a MIT style license - see LICENSE.rst

"""Define bit masks used in BOSS data and support symbolic operations on masks.
The SDSS bitmasks are documented at http://www.sdss3.org/dr10/algorithms/bitmasks.php.
The authoritative definition of the bit masks is the file
http://www.sdss3.org/svn/repo/idlutils/trunk/data/sdss/sdssMaskbits.par. A copy
of this file is included in this package's top-level directory and was used to
automatically generate the bitmask definitions in this file with the
:func:`extract_sdss_bitmasks` function.
"""

from __future__ import division, print_function


def define_bitmask(mask_name, mask_description, **bits):
    """Define a new type for a bitmask with specified symbolic bit names.
    After defining a bitmask type, its bit names are accessible as class-level
    attributes of the returned type and can be used as integer values, for example::
        >>> COLORS = define_bitmask('COLORS','Primary Colors',RED=0,BLUE=1,GREEN=4)
        >>> COLORS.BLUE
        2
        >>> '{0:b}'.format(COLORS.RED|COLORS.GREEN)
        '10001'
    The :func:`decode_bitmask` function is useful for converting an integral value back
    to a list of symbolic bit names.
    Args:
        mask_name(str): The type name for this mask. By convention, this name is upper
            case and matches the name assigned to this function's return value, as
            in the examples above.
        mask_description(str): A description of this bit mask that will be available as
            the docstring of the new defined type.
        bits(dict): A dictionary of name,definition pairs that define the mapping from
            symbolic bit names to bit offsets and optional comments. Although
            this argument can be passed as a dictionary, the dictionary is usually
            implicitly defined by the argument list, as in the examples above.
            By convention, bit names are all upper case. Each bit definition can either
            be specified as an integer offset >= 0 or else an (offset,description) tuple.
    Returns:
        type: A new type with the specified name that has class-level attributes for
            each named bit (see the examples above).  The type also defines a reverse
            map that is normally accessed via :func:`decode_bitmask`.
    Raises:
        TypeError: missing name and/or description args.
        ValueError: bit definition is invalid or an offset is repeated.
    """
    # Scan the bit definitions.
    bit_names = []
    bit_offsets = []
    bit_descriptions = []
    for name, bit_definition in bits.items():
        bit_names.append(name)
        if isinstance(bit_definition, tuple):
            bit_offset, bit_description = bit_definition
            bit_offsets.append(int(bit_offset))
            bit_descriptions.append(bit_description)
        else:
            bit_offsets.append(int(bit_definition))
            bit_descriptions.append('')
    # Check that bit offset values are unique.
    if len(bit_offsets) != len(set(bit_offsets)):
        raise ValueError('Bit offset values must be unique.')
    # Initialize a class dictionary with attributes for each named bit.
    class_dict = dict(zip(bit_names, (1 << offset for offset in bit_offsets)))
    # Add a reverse-lookup mapping from offsets to names.
    class_dict['_reverse_map'] = dict(zip(bit_offsets, bit_names))
    # Add a dictionary of bit descriptions indexed by offset.
    class_dict['_description'] = dict(zip(bit_offsets, bit_descriptions))
    # Use the mask description as the first line of the docstring.
    docstring = mask_description
    # Append each bit definition as a list of google-sphinx style attributes.
    docstring += '\n\nAttributes:\n'
    for name, offset, description in zip(bit_names, bit_offsets, bit_descriptions):
        docstring += '    {0} (int): (1<<{1}) {2}\n'.format(name, offset, description)
    class_dict['__doc__'] = docstring
    # Return a new class type for this bitmask.
    return type(mask_name, (), class_dict)


def summarize_bitmask_values(mask, values, strict=True):
    """Summarize an array of bitmask values.
    Args:
        mask: A bitmask type, normally created with :func:`create_bitmask`, that defines
            the symbolic bit names to summarize.
        values(numpy.ndarray): An array of values that will be decoded and summarized.
    Returns:
        dict: A dictionary with bit names as keys and the number of values in which each
            bit is set as values.  Any bit that is never set will not appear in the list
            of keys.
    """
    summary = {}
    for value in values:
        for bit_name in decode_bitmask(mask, value, strict=strict):
            bit_count = summary.get(bit_name, 0)
            bit_count += 1
            summary[bit_name] = bit_count
    return summary


def decode_bitmask(mask, value, strict=True):
    """Decode a integer value into its symbolic bit names.
    Use this function to convert a bitmask value into a list of symbolic bit names,
    for example::
        >>> COLORS = define_bitmask('COLORS','Primary colors',RED=0,BLUE=1,GREEN=4)
        >>> decode_bitmask(COLORS,COLORS.RED|COLORS.BLUE)
        ('RED', 'BLUE')
    For pretty printing, try::
        >>> print('|'.join(decode_bitmask(COLORS,COLORS.RED|COLORS.BLUE)))
        RED|BLUE
    Args:
        mask: A bitmask type, normally created with :func:`create_bitmask`, that defines
            the symbolic bit names to use for the decoding.
        value(int): The integral value to decode.
        strict(bool): If set, then all bits set in value must be defined in the bitmask
            type definition.
    Returns:
        tuple: A tuple of symbolic bit names in order of increasing bit offset. If strict
            is False, then any bits without corresponding symbolic names will appear as
            '1<<n' for offset n.
    Raises:
        AttributeError: mask does not have the attributes necessary to define a bitmask.
        ValueError: value has a bit set that has no symbolic name defined and strict is True.
    """
    names = []
    shifted = int(value)
    offset = 0
    while shifted:
        if shifted & 1:
            name = mask._reverse_map.get(offset)
            if name is None:
                if strict:
                    raise ValueError('Bit {0:d} is set but has no symbolic name defined.'.format(offset))
                name = '1<<{0:d}'.format(offset)
            names.append(name)
        offset += 1
        shifted = shifted >> 1
    return tuple(names)


def bitmask_from_text(mask, text):
    """Initialize a bitmask from text.
    Builds an integer value from text containing bit names that should be set. The
    complement of :func:`decode_bitmask`. For example::
        >>> COLORS = define_bitmask('COLORS','Primary colors',RED=0,BLUE=1,GREEN=4)
        >>> '{0:b}'.format(bitmask_from_text(COLORS,'GREEN|BLUE'))
        '10010'
    Args:
        mask: A bitmask type, normally created with :func:`create_bitmask`, that defines
            the symbolic bit names that are allowed.
        text: A list of bit names separated by '|'.
    Returns:
        int: Integer with bits set for each bit name appearing in the text.
    Raises:
        ValueError: invalid text specification.
    """
    if not hasattr(mask, '__dict__'):
        raise ValueError('Invalid bitmask.')
    value = int(0)
    for bit_name in text.split('|'):
        if bit_name not in mask.__dict__:
            raise ValueError('Invalid bit name: {0}.'.format(bit_name))
        value = value | mask.__dict__[bit_name]
    return value


def extract_sdss_bitmasks(filename='sdssMaskbits.par', indent=' ' * 4):
    """Scan the parfile defining SDSS bitmasks and print code to define these types for bossdata.bits.
    This function is intended to be run by hand with the output pasted into this module, to
    bootstrap or update the official SDSS bitmask definitions defined here. The generated code
    is printed directly to the standard output. This function should normally be run from the
    package top-level directory as::
        python bossdata/bits.py > bitdefs.py
    and will read `sdssMaskBits.par` from the same directory.  The contents of `bitdefs.py` is
    then pasted directly into this file, replacing any previous pasted version.
    Args:
        filename(str): Path of the parfile to read.
        indent(str): Indentation to use in the generated output.
    Raises:
        RuntimeError: Parse error while reading the input file.
    """

    import shlex

    mask_name = None
    with open(filename) as f:
        for line_number, line in enumerate(f):
            # Skip comment lines
            if line[0] == '#':
                continue
            # Parse the line into tokens, respecting double-quoted strings.
            try:
                tokens = shlex.split(line)
            except ValueError as e:
                raise RuntimeError('Parse error on line {0:d}: {1:s}'.format(line_number, e))
            if not tokens:
                continue

            if tokens[0] == 'masktype':
                # End any previous definition.
                if mask_name:
                    print(')')
                # Start a new definition.
                mask_name, description = tokens[1], tokens[3]
                print('{0} = define_bitmask(\n{1}"{0}",\n{1}"{2}",'.format(
                    mask_name, indent, description))

            if tokens[0] == 'maskbits':
                if tokens[1] != mask_name:
                    raise RuntimeError('Parsing error on line {0:d}: found {1} but expected {2}.'.format(
                        line_number, tokens[1], mask_name))
                offset, name, description = tokens[2:5]
                # prepend an underscore to any names beginning with a digit so they are valid python names.
                if name[0].isdigit():
                    name = '_' + name
                print('{0}{1:30s} = ({2:3d}, "{3}"),'.format(indent, name, int(offset), description))

        # End the final mask definition.
        if mask_name:
            print(')')

# The bitmask definitions below were automatically generated using the extract_sdss_bitmasks function
# and should not be edited by hand.
SPPIXMASK = define_bitmask(
    "SPPIXMASK",
    "Mask bits for an SDSS spectrum. 0-15 refer to each fiber, 16-31 refer to each pixel in a spectrum.",
    NOPLUG                         = (  0, "Fiber not listed in plugmap file"),
    BADTRACE                       = (  1, "Bad trace from routine TRACE320CRUDE"),
    BADFLAT                        = (  2, "Low counts in fiberflat"),
    BADARC                         = (  3, "Bad arc solution"),
    MANYBADCOLUMNS                 = (  4, "More than 10% of pixels are bad columns"),
    MANYREJECTED                   = (  5, "More than 10% of pixels are rejected in extraction"),
    LARGESHIFT                     = (  6, "Large spatial shift between flat and object position"),
    BADSKYFIBER                    = (  7, "Sky fiber shows extreme residuals"),
    NEARWHOPPER                    = (  8, "Within 2 fibers of a whopping fiber (exclusive)"),
    WHOPPER                        = (  9, "Whopping fiber, with a very bright source."),
    SMEARIMAGE                     = ( 10, "Smear available for red and blue cameras"),
    SMEARHIGHSN                    = ( 11, "S/N sufficient for full smear fit"),
    SMEARMEDSN                     = ( 12, "S/N only sufficient for scaled median fit"),
    NEARBADPIXEL                   = ( 16, "Bad pixel within 3 pixels of trace."),
    LOWFLAT                        = ( 17, "Flat field less than 0.5"),
    FULLREJECT                     = ( 18, "Pixel fully rejected in extraction (INVVAR=0)"),
    PARTIALREJECT                  = ( 19, "Some pixels rejected in extraction"),
    SCATTEREDLIGHT                 = ( 20, "Scattered light significant"),
    CROSSTALK                      = ( 21, "Cross-talk significant"),
    NOSKY                          = ( 22, "Sky level unknown at this wavelength (INVVAR=0)"),
    BRIGHTSKY                      = ( 23, "Sky level > flux + 10*(flux_err) AND sky > 1.25 * median(sky,99 pixels)"),
    NODATA                         = ( 24, "No data available in combine B-spline (INVVAR=0)"),
    COMBINEREJ                     = ( 25, "Rejected in combine B-spline"),
    BADFLUXFACTOR                  = ( 26, "Low flux-calibration or flux-correction factor"),
    BADSKYCHI                      = ( 27, "Relative chi^2 > 3 in sky residuals at this wavelength"),
    REDMONSTER                     = ( 28, "Contiguous region of bad chi^2 in sky residuals (with threshhold of relative chi^2 > 3)."),
)
TARGET = define_bitmask(
    "TARGET",
    "Primary target mask bits in SDSS-I, -II (for LEGACY_TARGET1 or PRIMTARGET).",
    QSO_HIZ                        = (  0, "High-redshift (griz) QSO target"),
    QSO_CAP                        = (  1, "ugri-selected quasar at high Galactic latitude"),
    QSO_SKIRT                      = (  2, "ugri-selected quasar at low Galactic latitude"),
    QSO_FIRST_CAP                  = (  3, "FIRST source with stellar colors at high Galactic latitude"),
    QSO_FIRST_SKIRT                = (  4, "FIRST source with stellar colors at low Galactic latitude"),
    GALAXY_RED                     = (  5, "Luminous Red Galaxy target (any criteria)"),
    GALAXY                         = (  6, "Main sample galaxy"),
    GALAXY_BIG                     = (  7, "Low-surface brightness main sample galaxy (mu50>23 in r-band)"),
    GALAXY_BRIGHT_CORE             = (  8, "Galaxy targets who fail all the surface brightness selection limits but have r-band fiber magnitudes brighter than 19"),
    ROSAT_A                        = (  9, "ROSAT All-Sky Survey match, also a radio source"),
    ROSAT_B                        = ( 10, "ROSAT All-Sky Survey match, have SDSS colors of AGNs or quasars"),
    ROSAT_C                        = ( 11, "ROSAT All-Sky Survey match, fall in a broad intermediate category that includes stars that are bright, moderately blue, or both"),
    ROSAT_D                        = ( 12, "ROSAT All-Sky Survey match, are otherwise bright enough for SDSS spectroscopy"),
    STAR_BHB                       = ( 13, "blue horizontal-branch stars"),
    STAR_CARBON                    = ( 14, "dwarf and giant carbon stars"),
    STAR_BROWN_DWARF               = ( 15, "brown dwarfs (note this sample is tiled)"),
    STAR_SUB_DWARF                 = ( 16, "low-luminosity subdwarfs"),
    STAR_CATY_VAR                  = ( 17, "cataclysmic variables"),
    STAR_RED_DWARF                 = ( 18, "red dwarfs"),
    STAR_WHITE_DWARF               = ( 19, "hot white dwarfs"),
    SERENDIP_BLUE                  = ( 20, "lying outside the stellar locus in color space"),
    SERENDIP_FIRST                 = ( 21, "coincident with FIRST sources but fainter than the equivalent in quasar target selection (also includes non-PSF sources"),
    SERENDIP_RED                   = ( 22, "lying outside the stellar locus in color space"),
    SERENDIP_DISTANT               = ( 23, "lying outside the stellar locus in color space"),
    SERENDIP_MANUAL                = ( 24, "manual serendipity flag"),
    QSO_MAG_OUTLIER                = ( 25, "Stellar outlier; too faint or too bright to be targeted"),
    GALAXY_RED_II                  = ( 26, "Luminous Red Galaxy target (Cut II criteria)"),
    ROSAT_E                        = ( 27, "ROSAT All-Sky Survey match, but too faint or too bright for SDSS spectroscopy"),
    STAR_PN                        = ( 28, "central stars of planetary nebulae"),
    QSO_REJECT                     = ( 29, "Object in explicitly excluded region of color space, therefore not targeted at QSO"),
    SOUTHERN_SURVEY                = ( 31, "Set in primtarget if this is a special program target"),
)
TTARGET = define_bitmask(
    "TTARGET",
    "Secondary target mask bits in SDSS-I, -II (for LEGACY_TARGET2, SPECIAL_TARGET2 or SECTARGET).",
    LIGHT_TRAP                     = (  0, "hole drilled for bright star, to avoid scattered light"),
    REDDEN_STD                     = (  1, "reddening standard star"),
    TEST_TARGET                    = (  2, "a test target"),
    QA                             = (  3, "quality assurance target"),
    SKY                            = (  4, "sky target"),
    SPECTROPHOTO_STD               = (  5, "spectrophotometry standard (typically an F-star)"),
    GUIDE_STAR                     = (  6, "guide star hole"),
    BUNDLE_HOLE                    = (  7, "fiber bundle hole"),
    QUALITY_HOLE                   = (  8, "hole drilled for plate shop quality measurements"),
    HOT_STD                        = (  9, "hot standard star"),
    SOUTHERN_SURVEY                = ( 31, "a segue or southern survey target"),
)
ZWARNING = define_bitmask(
    "ZWARNING",
    "Warnings for SDSS spectra.",
    SKY                            = (  0, "sky fiber"),
    LITTLE_COVERAGE                = (  1, "too little wavelength coverage (WCOVERAGE < 0.18)"),
    SMALL_DELTA_CHI2               = (  2, "chi-squared of best fit is too close to that of second best (<0.01 in reduced chi-sqaured)"),
    NEGATIVE_MODEL                 = (  3, "synthetic spectrum is negative (only set for stars and QSOs)"),
    MANY_OUTLIERS                  = (  4, "fraction of points more than 5 sigma away from best model is too large (>0.05)"),
    Z_FITLIMIT                     = (  5, "chi-squared minimum at edge of the redshift fitting range (Z_ERR set to -1)"),
    NEGATIVE_EMISSION              = (  6, "a QSO line exhibits negative emission, triggered only in QSO spectra, if  C_IV, C_III, Mg_II, H_beta, or H_alpha has LINEAREA + 3 * LINEAREA_ERR < 0"),
    UNPLUGGED                      = (  7, "the fiber was unplugged, so no spectrum obtained"),
    BAD_TARGET                     = (  8, "catastrophically bad targeting data (e.g. ASTROMBAD in CALIB_STATUS)"),
    NODATA                         = (  9, "No data for this fiber, e.g. because spectrograph was broken during this exposure (ivar=0 for all pixels)"),
)
RESOLVE_STATUS = define_bitmask(
    "RESOLVE_STATUS",
    "Resolve status for an SDSS catalog entry.  Only one of bits RUN_PRIMARY, RUN_RAMP, RUN_OVERLAPONLY, RUN_IGNORE, and RUN_DUPLICATE can be set. RUN_EDGE can be set for any object. To get a unique set of objects across the whole survey, search for objects with SURVEY_PRIMARY set. To get a unique set of objects within a run, search for objects with RUN_PRIMARY set.",
    RUN_PRIMARY                    = (  0, "primary within the objects own run (but not necessarily for the survey as a whole)"),
    RUN_RAMP                       = (  1, "in what would be the overlap area of a field, but with no neighboring field"),
    RUN_OVERLAPONLY                = (  2, "only appears in the overlap between two fields"),
    RUN_IGNORE                     = (  3, "bright or parent object that should be ignored"),
    RUN_EDGE                       = (  4, "near lowest or highest column"),
    RUN_DUPLICATE                  = (  5, "duplicate measurement of same pixels in two different fields"),
    SURVEY_PRIMARY                 = (  8, "Primary observation within the full survey, where it appears in the primary observation of this part of the sky"),
    SURVEY_BEST                    = (  9, "Best observation within the full survey, but it does not appear in the primary observation of this part of the sky"),
    SURVEY_SECONDARY               = ( 10, "Repeat (independent) observation of an object that has a different primary or best observation"),
    SURVEY_BADFIELD                = ( 11, "In field with score=0"),
    SURVEY_EDGE                    = ( 12, "Not kept as secondary because it is RUN_RAMP or RUN_EDGE object"),
)
IMAGE_STATUS = define_bitmask(
    "IMAGE_STATUS",
    "Sky and instrument conditions of SDSS image",
    CLEAR                          = (  0, "Clear skies"),
    CLOUDY                         = (  1, "Cloudy skies (unphotometric)"),
    UNKNOWN                        = (  2, "Sky conditions unknown (unphotometric)"),
    BAD_ROTATOR                    = (  3, "Rotator problems (set score=0)"),
    BAD_ASTROM                     = (  4, "Astrometry problems (set score=0)"),
    BAD_FOCUS                      = (  5, "Focus bad (set score=0)"),
    SHUTTERS                       = (  6, "Shutter out of place (set score=0)"),
    FF_PETALS                      = (  7, "Flat-field petals out of place (unphotometric)"),
    DEAD_CCD                       = (  8, "CCD bad (unphotometric)"),
    NOISY_CCD                      = (  9, "CCD noisy (unphotometric)"),
)
CALIB_STATUS = define_bitmask(
    "CALIB_STATUS",
    "Calibration status for an SDSS image",
    PHOTOMETRIC                    = (  0, "Photometric observations"),
    UNPHOT_OVERLAP                 = (  1, "Unphotometric observations, calibrated based on overlaps with clear, ubercalibrated data; this is done on a field-by-field basis"),
    UNPHOT_EXTRAP_CLEAR            = (  2, "Extrapolate the solution from the clear part of a night (that was ubercalibrated) to the cloudy part"),
    UNPHOT_EXTRAP_CLOUDY           = (  3, "   The solution here is based on fitting the a-term to cloudy data."),
    UNPHOT_DISJOINT                = (  4, "Data is disjoint from the rest of the survey (even though conditions may be photometric), the calibration is suspect"),
    INCREMENT_CALIB                = (  5, "Incrementally calibrated by considering overlaps with ubercalibrated data"),
    PS1_UNPHOT                     = (  6, "Comparison to PS1 reveals unphotometric conditions"),
    PS1_CONTRAIL                   = (  7, "Comparison to PS1 reveals possible contrail"),
    PT_CLEAR                       = (  8, "PT calibration for clear data"),
    PT_CLOUDY                      = (  9, "PT calibration for cloudy data"),
    DEFAULT                        = ( 10, "a default calibration used"),
    NO_UBERCAL                     = ( 11, "not uber-calibrated"),
    ASTROMBAD                      = ( 12, "catastrophically bad astrometry"),
    PS1_PCOMP_MODEL                = ( 13, "Enough information for PS1-based principal component flat model"),
    PS1_LOW_RMS                    = ( 14, "Low RMS in comparison with PS1"),
)
VAGC_SELECT = define_bitmask(
    "VAGC_SELECT",
    "Selection flags for Main VAGC sample",
    TILED                          = (  0, "selected because near a tiled target"),
    PLATEHOLE                      = (  1, "selected because near a hole on an SDSS plate"),
    MAIN                           = (  2, "selected according to slightly adjusted Main sample criteria"),
)
OBJECT1 = define_bitmask(
    "OBJECT1",
    "Object flags from photo reductions for SDSS (first 32)",
    CANONICAL_CENTER               = (  0, "The quantities (psf counts, model fits and likelihoods) that are usually determined at an object's center as determined band-by-band were in fact determined at the canonical center (suitably transformed). This is due to the object being to close to the edge to extract a profile at the local center, and OBJECT1_EDGE is also set."),
    BRIGHT                         = (  1, "Indicates that the object was detected as a bright object. Since these are typically remeasured as faint objects, most users can ignore BRIGHT objects."),
    EDGE                           = (  2, "Object is too close to edge of frame in this band."),
    BLENDED                        = (  3, "Object was determined to be a blend. The flag is set if: more than one peak is detected within an object in a single band together; distinct peaks are found when merging different colours of one object together; or distinct peaks result when merging different objects together. "),
    CHILD                          = (  4, "Object is a child, created by the deblender."),
    PEAKCENTER                     = (  5, "Given center is position of peak pixel, as attempts to determine a better centroid failed."),
    NODEBLEND                      = (  6, "Although this object was marked as a blend, no deblending was attempted."),
    NOPROFILE                      = (  7, "Frames couldn't extract a radial profile."),
    NOPETRO                        = (  8, " No Petrosian radius or other Petrosian quanties could be measured."),
    MANYPETRO                      = (  9, "Object has more than one possible Petrosian radius."),
    NOPETRO_BIG                    = ( 10, "The Petrosian ratio has not fallen to the value at which the Petrosian radius is defined at the outermost point of the extracted radial profile. NOPETRO is set, and the Petrosian radius is set to the outermost point in the profile."),
    DEBLEND_TOO_MANY_PEAKS         = ( 11, "The object had the OBJECT1_DEBLEND flag set, but it contained too many candidate children to be fully deblended. This flag is only set in the parent, i.e. the object with too many peaks."),
    CR                             = ( 12, "Object contains at least one pixel which was contaminated by a cosmic ray. The OBJECT1_INTERP flag is also set. This flag does not mean that this object is a cosmic ray; rather it means that a cosmic ray has been removed. "),
    MANYR50                        = ( 13, " More than one radius was found to contain 50% of the Petrosian flux. (For this to happen part of the radial profile must be negative)."),
    MANYR90                        = ( 14, "More than one radius was found to contain 90% of the Petrosian flux. (For this to happen part of the radial profile must be negative)."),
    BAD_RADIAL                     = ( 15, " Measured profile includes points with a S/N <= 0.  In practice this flag is essentially meaningless."),
    INCOMPLETE_PROFILE             = ( 16, "A circle, centerd on the object, of radius the canonical Petrosian radius extends beyond the edge of the frame. The radial profile is still measured from those parts of the object that do lie on the frame."),
    INTERP                         = ( 17, " The object contains interpolated pixels (e.g. cosmic rays or bad columns)."),
    SATUR                          = ( 18, "The object contains saturated pixels; INTERP is also set."),
    NOTCHECKED                     = ( 19, "Object includes pixels that were not checked for peaks, for example the unsmoothed edges of frames, and the cores of subtracted or saturated stars."),
    SUBTRACTED                     = ( 20, "Object (presumably a star) had wings subtracted."),
    NOSTOKES                       = ( 21, "Object has no measured Stokes parameters."),
    BADSKY                         = ( 22, "The estimated sky level is so bad that the central value of the radial profile is crazily negative; this is usually the result of the subtraction of the wings of bright stars failing."),
    PETROFAINT                     = ( 23, "At least one candidate Petrosian radius occured at an unacceptably low surface brightness."),
    TOO_LARGE                      = ( 24, " The object is (as it says) too large.  Either the object is still detectable at the outermost point of the extracted radial profile (a radius of approximately 260 arcsec), or when attempting to deblend an object, at least one child is larger than half a frame (in either row or column)."),
    DEBLENDED_AS_PSF               = ( 25, "When deblending an object, in this band this child was treated as a PSF."),
    DEBLEND_PRUNED                 = ( 26, "When solving for the weights to be assigned to each child the deblender encountered a nearly singular matrix, and therefore deleted at least one of them."),
    ELLIPFAINT                     = ( 27, "No isophotal fits were performed."),
    BINNED1                        = ( 28, "The object was detected in an unbinned image."),
    BINNED2                        = ( 29, " The object was detected in a 2x2 binned image after all unbinned detections have been replaced by the background level."),
    BINNED4                        = ( 30, "The object was detected in a 4x4 binned image. The objects detected in the 2x2 binned image are not removed before doing this."),
    MOVED                          = ( 31, "The object appears to have moved during the exposure. Such objects are candidates to be deblended as moving objects."),
)
OBJECT2 = define_bitmask(
    "OBJECT2",
    "Object flags from photo reductions for SDSS (second 32)",
    DEBLENDED_AS_MOVING            = (  0, "The object has the MOVED flag set, and was deblended on the assumption that it was moving."),
    NODEBLEND_MOVING               = (  1, "The object has the MOVED flag set, but was not deblended as a moving object."),
    TOO_FEW_DETECTIONS             = (  2, "The object has the MOVED flag set, but has too few detection to be deblended as moving."),
    BAD_MOVING_FIT                 = (  3, "The fit to the object as a moving object is too bad to be believed."),
    STATIONARY                     = (  4, "A moving objects velocity is consistent with zero"),
    PEAKS_TOO_CLOSE                = (  5, "Peaks in object were too close (set only in parent objects)."),
    BINNED_CENTER                  = (  6, "When centroiding the object the object's size is larger than the (PSF) filter used to smooth the image."),
    LOCAL_EDGE                     = (  7, "The object's center in some band was too close to the edge of the frame to extract a profile."),
    BAD_COUNTS_ERROR               = (  8, "An object containing interpolated pixels had too few good pixels to form a reliable estimate of its error"),
    BAD_MOVING_FIT_CHILD           = (  9, "A putative moving child's velocity fit was too poor, so it was discarded, and the parent was not deblended as moving"),
    DEBLEND_UNASSIGNED_FLUX        = ( 10, "After deblending, the fraction of flux assigned to none of the children was too large (this flux is then shared out as described elsewhere)."),
    SATUR_CENTER                   = ( 11, "An object's center is very close to at least one saturated pixel; the object may well be causing the saturation."),
    INTERP_CENTER                  = ( 12, "An object's center is very close to at least one interpolated pixel."),
    DEBLENDED_AT_EDGE              = ( 13, "An object so close to the edge of the frame that it would not ordinarily be deblended has been deblended anyway.  Only set for objects large enough to be EDGE in all fields/strips."),
    DEBLEND_NOPEAK                 = ( 14, "A child had no detected peak in a given band, but we centroided it anyway and set the BINNED1"),
    PSF_FLUX_INTERP                = ( 15, "The fraction of light actually detected (as opposed to guessed at by the interpolator) was less than some number (currently 80%) of the total."),
    TOO_FEW_GOOD_DETECTIONS        = ( 16, "A child of this object had too few good detections to be deblended as moving."),
    CENTER_OFF_AIMAGE              = ( 17, "At least one peak's center lay off the atlas image in some band. This can happen when the object's being deblended as moving, or if the astrometry is badly confused."),
    DEBLEND_DEGENERATE             = ( 18, "At least one potential child has been pruned because its template was too similar to some other child's template."),
    BRIGHTEST_GALAXY_CHILD         = ( 19, "This is the brightest child galaxy in a blend."),
    CANONICAL_BAND                 = ( 20, "This band was the canonical band. This is the band used to measure the Petrosian radius used to calculate the Petrosian counts in each band, and to define the model used to calculate model colors; it has no effect upon the coordinate system used for the OBJC center."),
    AMOMENT_UNWEIGHTED             = ( 21, "`Adaptive' moments are actually unweighted."),
    AMOMENT_SHIFT                  = ( 22, "Object's center moved too far while determining adaptive moments. In this case, the M_e1 and M_e2 give the (row, column) shift, not the object's shape."),
    AMOMENT_MAXITER                = ( 23, "Too many iterations while determining adaptive moments."),
    MAYBE_CR                       = ( 24, "This object may be a cosmic ray.  This bit can get set in the cores of bright stars, and is quite likely to be set for the cores of saturated stars."),
    MAYBE_EGHOST                   = ( 25, "Object appears in the right place to be an electronics ghost."),
    NOTCHECKED_CENTER              = ( 26, "Center of object lies in a NOTCHECKED region. The object is almost certainly bogus."),
    HAS_SATUR_DN                   = ( 27, "This object is saturated in this band and the bleed trail doesn't touch the edge of the frame, we we've made an attempt to add up all the flux in the bleed trails, and to include it in the object's photometry. "),
    DEBLEND_PEEPHOLE               = ( 28, "The deblend was modified by the optimizer"),
    SPARE3                         = ( 29, ""),
    SPARE2                         = ( 30, ""),
    SPARE1                         = ( 31, ""),
)
Q_EYEBALL = define_bitmask(
    "Q_EYEBALL",
    "Quality eyeball flags from VAGC",
    DONE                           = (  0, ""),
    OTHER                          = (  1, ""),
    UNCLASSIFIABLE                 = (  2, ""),
    NEED_BIGGER_IMAGE              = (  3, ""),
    BAD_DEBLEND                    = (  4, ""),
    FLECK                          = (  5, ""),
    DOUBLE_STAR                    = (  6, ""),
    HII                            = (  7, ""),
    USE_ANYWAY                     = (  8, ""),
    EDGE                           = (  9, ""),
    SATELLITE                      = ( 10, ""),
    PLANE                          = ( 11, ""),
    BAD_Z                          = ( 12, ""),
    INTERNAL_REFLECTION            = ( 13, ""),
    BAD_SPEC_CLASS                 = ( 14, ""),
    USE_PARENT                     = ( 15, ""),
    IN_HUGE_OBJECT                 = ( 16, ""),
    STAR_ON_GALAXY                 = ( 17, ""),
    QSO_ON_GALAXY                  = ( 18, ""),
    NEGATIVE_QSO_FIT               = ( 19, ""),
    BAD_SPECTRUM                   = ( 20, ""),
    POSSIBLE_LENS                  = ( 21, ""),
    IS_STAR                        = ( 22, ""),
    DOUBLE_Z                       = ( 23, ""),
    PLANETARY_NEBULA               = ( 24, ""),
    BAD_PARENT_CENTER              = ( 25, ""),
    GOOD_Z                         = ( 26, ""),
    USE_CHILD_IMAGE                = ( 27, ""),
    USE_CHILD_SPECTRUM             = ( 28, ""),
)
T_EYEBALL = define_bitmask(
    "T_EYEBALL",
    "Type eyeball flags from VAGC",
    DONE                           = (  0, ""),
    OTHER                          = (  1, ""),
    UNCLASSIFIABLE                 = (  2, ""),
    ELLIPTICAL                     = (  3, ""),
    DISK                           = (  4, ""),
    IRREGULAR                      = (  5, ""),
    UNUSED_0                       = (  6, ""),
    S0                             = (  7, ""),
    PITCH_0                        = (  8, "tightly wound"),
    PITCH_1                        = (  9, ""),
    PITCH_2                        = ( 10, ""),
    PITCH_3                        = ( 11, ""),
    PITCH_4                        = ( 12, "openly wound"),
    PSF                            = ( 13, ""),
    ASYMMETRIC                     = ( 14, ""),
    HII_REGIONS                    = ( 15, ""),
    SPIRAL_STRUCTURE               = ( 16, ""),
    DUST_LANE                      = ( 17, ""),
    BAR                            = ( 18, ""),
    RING                           = ( 19, ""),
    TIDAL_TAILS                    = ( 20, ""),
    SHELLS                         = ( 21, ""),
    BLUE_CORE                      = ( 22, ""),
    WARPED_DISK                    = ( 23, ""),
    DUST_ASYMMETRY                 = ( 24, ""),
    NEAR_NEIGHBORS                 = ( 25, ""),
    MERGER                         = ( 26, ""),
    OUTFLOW                        = ( 27, ""),
)
M_EYEBALL = define_bitmask(
    "M_EYEBALL",
    "Eyeball flags for mergers in VAGC",
    DONE                           = (  0, ""),
    NOT_MERGER                     = (  1, ""),
    QUESTIONABLE                   = (  2, ""),
    DRY                            = (  3, ""),
    TIDAL_TAILS                    = (  4, ""),
    SHELLS                         = (  5, ""),
    RING                           = (  6, ""),
    MAJOR                          = (  7, ""),
    MULTIPLE                       = (  8, ""),
    ALL_RED                        = (  9, ""),
    ALL_BLUE                       = ( 10, ""),
    MIXED_REDBLUE                  = ( 11, ""),
    REPEAT                         = ( 12, ""),
    BEFORE                         = ( 13, ""),
    DURING                         = ( 14, ""),
    AFTER                          = ( 15, ""),
)
FLUXMATCH_STATUS = define_bitmask(
    "FLUXMATCH_STATUS",
    "Flags from flux-based matching to SDSS photometry",
    ORIGINAL_FLUXMATCH             = (  0, "used the original positional match (which exists)"),
    FIBER_FLUXMATCH                = (  1, "flagged due to fiberflux/aperflux issue"),
    NONMATCH_FLUXMATCH             = (  2, "flagged due to non-match"),
    NOPARENT_FLUXMATCH             = (  3, "no overlapping parent in primary field"),
    PARENT_FLUXMATCH               = (  4, "overlapping parent has no children, so used it"),
    BRIGHTEST_FLUXMATCH            = (  5, "picked the brightest child"),
)
BOSS_TARGET1 = define_bitmask(
    "BOSS_TARGET1",
    "BOSS survey primary target selection flags",
    GAL_LOZ                        = (  0, "low-z lrgs"),
    GAL_CMASS                      = (  1, "dperp > 0.55, color-mag cut "),
    GAL_CMASS_COMM                 = (  2, "dperp > 0.55, commissioning color-mag cut"),
    GAL_CMASS_SPARSE               = (  3, "GAL_CMASS_COMM & (!GAL_CMASS) & (i < 19.9) sparsely sampled"),
    SDSS_KNOWN                     = (  6, "Matches a known SDSS spectra"),
    GAL_CMASS_ALL                  = (  7, "GAL_CMASS and the entire sparsely sampled region"),
    GAL_IFIBER2_FAINT              = (  8, "ifiber2 > 21.5, extinction corrected.  Used after Nov 2010"),
    GAL_LODPERP_DEPRECATED         = (  5, "(DEPRECATED) Same as hiz but between dperp00 and dperp0"),
    QSO_CORE                       = ( 10, "restrictive qso selection: commissioning only"),
    QSO_BONUS                      = ( 11, "permissive qso selection:  commissioning only"),
    QSO_KNOWN_MIDZ                 = ( 12, "known qso between [2.2,9.99]"),
    QSO_KNOWN_LOHIZ                = ( 13, "known qso outside of miz range. never target"),
    QSO_NN                         = ( 14, "Neural Net that match to sweeps/pass cuts"),
    QSO_UKIDSS                     = ( 15, "UKIDSS stars that match sweeps/pass flag cuts"),
    QSO_KDE_COADD                  = ( 16, "kde targets from the stripe82 coadd "),
    QSO_LIKE                       = ( 17, "likelihood method"),
    QSO_FIRST_BOSS                 = ( 18, "FIRST radio match"),
    QSO_KDE                        = ( 19, "selected by kde+chi2"),
    QSO_CORE_MAIN                  = ( 40, "Main survey core sample"),
    QSO_BONUS_MAIN                 = ( 41, "Main survey bonus sample"),
    QSO_CORE_ED                    = ( 42, "Extreme Deconvolution in Core"),
    QSO_CORE_LIKE                  = ( 43, "Likelihood that make it into core"),
    QSO_KNOWN_SUPPZ                = ( 44, "known qso between [1.8,2.15]"),
    STD_FSTAR                      = ( 20, "standard f-stars"),
    STD_WD                         = ( 21, "white dwarfs"),
    STD_QSO                        = ( 22, "qso"),
    TEMPLATE_GAL_PHOTO             = ( 32, "galaxy templates "),
    TEMPLATE_QSO_SDSS1             = ( 33, "QSO templates "),
    TEMPLATE_STAR_PHOTO            = ( 34, "stellar templates "),
    TEMPLATE_STAR_SPECTRO          = ( 35, "stellar templates (spectroscopically known)"),
)
ANCILLARY_TARGET1 = define_bitmask(
    "ANCILLARY_TARGET1",
    "BOSS survey target flags for ancillary programs",
    AMC                            = (  0, "defined in blake_boss_v2.descr"),
    FLARE1                         = (  1, "defined in blake_boss_v2.descr"),
    FLARE2                         = (  2, "defined in blake_boss_v2.descr"),
    HPM                            = (  3, "defined in blake_boss_v2.descr"),
    LOW_MET                        = (  4, "defined in blake_boss_v2.descr"),
    VARS                           = (  5, "defined in blake_boss_v2.descr"),
    BLAZGVAR                       = (  6, "defined in brandtxmm-andersonblazar-merged.descr "),
    BLAZR                          = (  7, "defined in brandtxmm-andersonblazar-merged.descr "),
    BLAZXR                         = (  8, "defined in brandtxmm-andersonblazar-merged.descr "),
    BLAZXRSAM                      = (  9, "defined in brandtxmm-andersonblazar-merged.descr "),
    BLAZXRVAR                      = ( 10, "defined in brandtxmm-andersonblazar-merged.descr "),
    XMMBRIGHT                      = ( 11, "defined in brandtxmm-andersonblazar-merged.descr "),
    XMMGRIZ                        = ( 12, "defined in brandtxmm-andersonblazar-merged.descr "),
    XMMHR                          = ( 13, "defined in brandtxmm-andersonblazar-merged.descr "),
    XMMRED                         = ( 14, "defined in brandtxmm-andersonblazar-merged.descr "),
    FBQSBAL                        = ( 15, "defined in master-BAL-targets.descr"),
    LBQSBAL                        = ( 16, "defined in master-BAL-targets.descr"),
    ODDBAL                         = ( 17, "defined in master-BAL-targets.descr"),
    OTBAL                          = ( 18, "defined in master-BAL-targets.descr"),
    PREVBAL                        = ( 19, "defined in master-BAL-targets.descr"),
    VARBAL                         = ( 20, "defined in master-BAL-targets.descr"),
    BRIGHTGAL                      = ( 21, "defined in bright_gal_v3.descr"),
    QSO_AAL                        = ( 22, "defined in qsoals_v2.descr "),
    QSO_AALS                       = ( 23, "defined in qsoals_v2.descr "),
    QSO_IAL                        = ( 24, "defined in qsoals_v2.descr "),
    QSO_RADIO                      = ( 25, "defined in qsoals_v2.descr "),
    QSO_RADIO_AAL                  = ( 26, "defined in qsoals_v2.descr "),
    QSO_RADIO_IAL                  = ( 27, "defined in qsoals_v2.descr "),
    QSO_NOAALS                     = ( 28, "defined in qsoals_v2.descr "),
    QSO_GRI                        = ( 29, "defined in sdss3_fan.descr "),
    QSO_HIZ                        = ( 30, "defined in sdss3_fan.descr "),
    QSO_RIZ                        = ( 31, "defined in sdss3_fan.descr "),
    RQSS_SF                        = ( 32, "defined in rqss090630.descr"),
    RQSS_SFC                       = ( 33, "defined in rqss090630.descr"),
    RQSS_STM                       = ( 34, "defined in rqss090630.descr"),
    RQSS_STMC                      = ( 35, "defined in rqss090630.descr"),
    SN_GAL1                        = ( 36, "defined in ancillary_supernova_hosts_v5.descr"),
    SN_GAL2                        = ( 37, "defined in ancillary_supernova_hosts_v5.descr"),
    SN_GAL3                        = ( 38, "defined in ancillary_supernova_hosts_v5.descr"),
    SN_LOC                         = ( 39, "defined in ancillary_supernova_hosts_v5.descr"),
    SPEC_SN                        = ( 40, "defined in ancillary_supernova_hosts_v5.descr"),
    SPOKE                          = ( 41, "defined in BOSS_slowpokes_v2.descr"),
    WHITEDWARF_NEW                 = ( 42, "defined in WDv5_eisenste_fixed.descr"),
    WHITEDWARF_SDSS                = ( 43, "defined in WDv5_eisenste_fixed.descr"),
    BRIGHTERL                      = ( 44, "defined in sd3targets_final.descr"),
    BRIGHTERM                      = ( 45, "defined in sd3targets_final.descr"),
    FAINTERL                       = ( 46, "defined in sd3targets_final.descr"),
    FAINTERM                       = ( 47, "defined in sd3targets_final.descr"),
    RED_KG                         = ( 48, "defined in redkg.descr"),
    RVTEST                         = ( 49, "defined in redkg.descr"),
    BLAZGRFLAT                     = ( 50, "defined in      anderson-blazar.par"),
    BLAZGRQSO                      = ( 51, "defined in anderson-blazar.par "),
    BLAZGX                         = ( 52, "defined in anderson-blazar.par"),
    BLAZGXQSO                      = ( 53, "defined in anderson-blazar.par"),
    BLAZGXR                        = ( 54, "defined in anderson-blazar.par"),
    BLUE_RADIO                     = ( 56, "defined in tremonti-blue-radio.fits.gz"),
    CHANDRAV1                      = ( 57, "defined in haggard-sf-accrete.fits"),
    CXOBRIGHT                      = ( 58, "defined in brandt-xray.par"),
    CXOGRIZ                        = ( 59, "defined in brandt-xray.par"),
    CXORED                         = ( 60, "defined in brandt-xray.par"),
    ELG                            = ( 61, "defined in kneib-cfht-elg.fits"),
    GAL_NEAR_QSO                   = ( 62, "defined in weiner-qso-sightline.fits"),
    MTEMP                          = ( 63, "defined in blake-transient-v3.fits"),
)
ANCILLARY_TARGET2 = define_bitmask(
    "ANCILLARY_TARGET2",
    "additional BOSS survey target flags for ancillary programs",
    HIZQSO82                       = (  0, "defined in mcgreer-hizqso.fits"),
    HIZQSOIR                       = (  1, "defined in mcgreer-hizqso.fits"),
    KQSO_BOSS                      = (  2, "defined in mcmahon-ukidss.fits"),
    QSO_VAR                        = (  3, "defined in butler-variable.fits.gz"),
    QSO_VAR_FPG                    = (  4, "defined in nathalie-ancillary3.par"),
    RADIO_2LOBE_QSO                = (  5, "defined in kimball-radio-2lobe-qso.fits.gz"),
    STRIPE82BCG                    = (  6, "defined in alexie-bcgs.fits"),
    QSO_SUPPZ                      = (  7, "defined in qso_suppz.descr"),
    QSO_VAR_SDSS                   = (  8, "defined in VARQSO.descr"),
    QSO_WISE_SUPP                  = (  9, "defined in BOSS_QSO_targets_July_WISE.descr"),
    QSO_WISE_FULL_SKY              = ( 10, "defined in none"),
    XMMSDSS                        = ( 11, "defined in georgakakis.descr"),
    IAMASERS                       = ( 12, "defined in zaw.descr"),
    DISKEMITTER_REPEAT             = ( 13, "defined in shen.descr"),
    WISE_BOSS_QSO                  = ( 14, "defined in ross_wisebossqso.descr"),
    QSO_XD_KDE_PAIR                = ( 15, "defined in myers.descr"),
    CLUSTER_MEMBER                 = ( 16, "defined in finoguenov_auxBOSS.descr"),
    SPOKE2                         = ( 17, "defined in dhital.descr"),
    FAINT_ELG                      = ( 18, "defined in comparat.descr"),
    PTF_GAL                        = ( 19, "defined in kasliwal.descr"),
    QSO_STD                        = ( 20, "defined in margala.descr"),
    HIZ_LRG                        = ( 21, "defined in newman.descr"),
    LRG_ROUND3                     = ( 22, "defined in newman.descr"),
    WISE_COMPLETE                  = ( 23, "defined in weiner_wise.descr"),
    TDSS_PILOT                     = ( 24, "defined in GreenMerloni_MD01.descr"),
    SPIDERS_PILOT                  = ( 25, "defined in GreenMerloni_MD01.descr"),
    TDSS_SPIDERS_PILOT             = ( 26, "defined in GreenMerloni_MD01.descr"),
    QSO_VAR_LF                     = ( 27, "defined in palanque_str82.descr"),
    TDSS_PILOT_PM                  = ( 28, "defined in TDSS_SPIDERS_MD03.descr"),
    TDSS_PILOT_SNHOST              = ( 29, "defined in TDSS_SPIDERS_MD03.descr"),
    FAINT_HIZ_LRG                  = ( 30, "defined in newman_lrg_w3.descr"),
    QSO_EBOSS_W3_ADM               = ( 31, "defined in myers_eboss_qso_w3.descr"),
    XMM_PRIME                      = ( 32, "defined in georgekaksi_xmmxll.descr"),
    XMM_SECOND                     = ( 33, "defined in georgekaksi_xmmxll.descr"),
    SEQUELS_ELG                    = ( 34, "defined in sequels_elg.descr"),
    GES                            = ( 35, "defined in rockosi_ges_segue.descr"),
    SEGUE1                         = ( 36, "defined in rockosi_ges_segue.descr"),
    SEGUE2                         = ( 37, "defined in rockosi_ges_segue.descr"),
    SDSSFILLER                     = ( 38, "defined in rockosi_ges_segue.descr"),
    SEQUELS_ELG_LOWP               = ( 39, "defined in sequels_elg.descr"),
    _25ORI_WISE                    = ( 40, "defined in knapp_25ori.descr"),
    _25ORI_WISE_W3                 = ( 41, "defined in knapp_25ori.descr"),
    KOEKAP_STAR                    = ( 42, "defined in knapp_kappaori.descr"),
    KOE2023_STAR                   = ( 43, "defined in knapp_ngc2023.descr"),
    KOE2068_STAR                   = ( 44, "defined in knapp_ngc2068.descr"),
    KOE2023BSTAR                   = ( 45, "defined in knapp_ngc2023.descr"),
    KOE2068BSTAR                   = ( 46, "defined in knapp_ngc2068.descr"),
    KOEKAPBSTAR                    = ( 47, "defined in knapp_kappaori.descr"),
    COROTGESAPOG                   = ( 48, "defined in rocksi_ges_segue.descr"),
    COROTGES                       = ( 49, "defined in rocksi_ges_segue.descr"),
    APOGEE                         = ( 50, "defined in rocksi_ges_segue.descr"),
    _2MASSFILL                     = ( 51, "defined in rocksi_ges_segue.descr"),
    TAU_STAR                       = ( 52, "defined in knapp_taurus.descr"),
    SEQUELS_TARGET                 = ( 53, "any target in SEQUELS darktime program"),
    RM_TILE1                       = ( 54, "reverberation mapping, high priority"),
    RM_TILE2                       = ( 55, "reverberation mapping, low priority"),
    QSO_DEEP                       = ( 56, "DEEP QSO described in QSO_DEEP_LBG.descr"),
    LBG                            = ( 57, "LBG described in QSO_DEEP_LBG.descr"),
    ELAIS_N1_LOFAR                 = ( 58, "LOFAR-selected target"),
    ELAIS_N1_FIRST                 = ( 59, "LOFAR-selected target"),
    ELAIS_N1_GMRT_GARN             = ( 60, "LOFAR-selected target"),
    ELAIS_N1_GMRT_TAYLOR           = ( 61, "LOFAR-selected target"),
    ELAIS_N1_JVLA                  = ( 62, "LOFAR-selected target"),
)
BOSSTILE_STATUS = define_bitmask(
    "BOSSTILE_STATUS",
    "BOSS tiling code status bits",
    TILED                          = (  0, "assigned a fiber"),
    NAKED                          = (  1, "not in area covered by tiles"),
    BOSSTARGET                     = (  2, "in the high priority set of targets"),
    DECOLLIDED                     = (  3, "in the decollided set of high priority"),
    ANCILLARY                      = (  4, "in the lower priority, ancillary set"),
    POSSIBLE_KNOCKOUT              = (  5, "knocked out of at least one tile by BOSSTARGET"),
    IGNORE_PRIORITY                = (  6, "priority exceeds max (ANCILLARY only)"),
    TOOBRIGHT                      = (  7, "fibermag too bright "),
    BLUEFIBER                      = (  8, "allocate this object a blue fiber"),
    CENTERPOST                     = (  9, "92 arcsec collision with center post"),
    REPEAT                         = ( 10, "included on more than one tile"),
    FILLER                         = ( 11, "was a filler (not normal repeat)"),
    NOT_TILED_TARGET               = ( 12, "though in input file, not a tiled target"),
    OUT_OF_BOUNDS                  = ( 13, "outside bounds for this sort of target (for restricted QSO geometry, e.g.)"),
    BAD_CALIB_STATUS               = ( 14, "bad CALIB_STATUS"),
    PREVIOUS_CHUNK                 = ( 15, "included because not tiled in previous overlapping chunk"),
    KNOWN_OBJECT                   = ( 16, "galaxy has known redshift"),
    DUPLICATE                      = ( 17, "trimmed as a duplicate object (only checked if not trimmed for any other reason)"),
    DUPLICATE_PRIMARY              = ( 18, "has associated duplicate object that were trimmed (but this one is kept)"),
    DUPLICATE_TILED                = ( 19, "trimmed as a duplicate object, and its primary was tiled"),
    TOOFAINT                       = ( 20, "trimmed because it was fainter than the ifiber2mag limit"),
    SUPPLEMENTARY                  = ( 21, "supplementary targets tiles after the ancillaries (subset of ancillaries)"),
    ANCILLARY_ROUND2               = ( 22, "new ancillaries added June 2012 (tiled after old ancillaries)"),
    MIDLEVEL_PRIORITY              = ( 23, "targets (from ancillary list) tiled between gals and ancillaries"),
)
EBOSS_TARGET0 = define_bitmask(
    "EBOSS_TARGET0",
    "targeting bitmask for SEQUELS (eBOSS precursor)",
    DO_NOT_OBSERVE                 = (  0, "Don't put a fiber on this object"),
    LRG_IZW                        = (  1, "LRG selection in i/z/W plane"),
    LRG_RIW                        = (  2, "LRG selection in r/i/W plan with (i-z) cut"),
    QSO_EBOSS_CORE                 = ( 10, "QSOs in XDQSOz+WISE selection for clustering"),
    QSO_PTF                        = ( 11, "QSOs with variability in PTF imaging"),
    QSO_REOBS                      = ( 12, "QSOs from BOSS to be reobserved"),
    QSO_EBOSS_KDE                  = ( 13, "KDE-selected QSOs (sequels only)"),
    QSO_EBOSS_FIRST                = ( 14, "Objects with FIRST radio matches"),
    QSO_BAD_BOSS                   = ( 15, "QSOs from BOSS with bad spectra"),
    QSO_BOSS_TARGET                = ( 16, "Known TARGETS from BOSS with spectra"),
    QSO_SDSS_TARGET                = ( 17, "Known TARGETS from SDSS with spectra"),
    QSO_KNOWN                      = ( 18, "Known QSOs from previous surveys"),
    DR9_CALIB_TARGET               = ( 19, "Target found in DR9-calibrated imaging"),
    SPIDERS_RASS_AGN               = ( 20, "RASS AGN sources"),
    SPIDERS_RASS_CLUS              = ( 21, "RASS Cluster sources"),
    SPIDERS_ERASS_AGN              = ( 22, "ERASS AGN sources"),
    SPIDERS_ERASS_CLUS             = ( 23, "ERASS Cluster sources"),
    TDSS_A                         = ( 30, "Main PanSTARRS selection for TDSS"),
    TDSS_FES_DE                    = ( 31, "TDSS Few epoch spectroscopy"),
    TDSS_FES_DWARFC                = ( 32, "TDSS Few epoch spectroscopy"),
    TDSS_FES_NQHISN                = ( 33, "TDSS Few epoch spectroscopy"),
    TDSS_FES_MGII                  = ( 34, "TDSS Few epoch spectroscopy"),
    TDSS_FES_VARBAL                = ( 35, "TDSS Few epoch spectroscopy"),
    SEQUELS_PTF_VARIABLE           = ( 40, "Variability objects from PTF"),
    SEQUELS_COLLIDED               = ( 41, "Collided galaxies from BOSS"),
)
EBOSS_TARGET1 = define_bitmask(
    "EBOSS_TARGET1",
    "targeting bitmask for eBOSS",
    DO_NOT_OBSERVE                 = (  0, "Don't put a fiber on this object"),
    LRG1_WISE                      = (  1, "LRG selection in i/z/W plane"),
    LRG1_IDROP                     = (  2, "LRG selection in r/i/W plan with (i-z) cut"),
    LRG_KNOWN                      = (  3, "LRG selection in r/i/W plan with (i-z) cut"),
    QSO1_VAR_S82                   = (  9, "Variability-selected QSOs in the repeated Stripe 82 imaging"),
    QSO1_EBOSS_CORE                = ( 10, "QSOs in XDQSOz+WISE selection for clustering"),
    QSO1_PTF                       = ( 11, "QSOs with variability in PTF imaging"),
    QSO1_REOBS                     = ( 12, "QSOs from BOSS to be reobserved"),
    QSO1_EBOSS_KDE                 = ( 13, "KDE-selected QSOs (sequels only)"),
    QSO1_EBOSS_FIRST               = ( 14, "Ojbects with FIRST radio matches"),
    QSO1_BAD_BOSS                  = ( 15, "QSOs from BOSS with bad spectra"),
    QSO_BOSS_TARGET                = ( 16, "Known TARGETS from BOSS with spectra"),
    QSO_SDSS_TARGET                = ( 17, "Known TARGETS from SDSS with spectra"),
    QSO_KNOWN                      = ( 18, "Known QSOs from previous surveys"),
    TDSS_TARGET                    = ( 30, "Target for TDSS (subclass found in eboss_target2)"),
    SPIDERS_TARGET                 = ( 31, "Target for SPIDERS (subclass found in eboss_target2)"),
    ELG_TEST1                      = ( 40, "Test targets for ELG selection"),
    STD_FSTAR                      = ( 50, "standard f-stars"),
    STD_WD                         = ( 51, "white dwarfs"),
    STD_QSO                        = ( 52, "qso"),
)
EBOSS_TARGET2 = define_bitmask(
    "EBOSS_TARGET2",
    "targeting bitmask for eBOSS",
    SPIDERS_RASS_AGN               = (  0, "RASS AGN sources"),
    SPIDERS_RASS_CLUS              = (  1, "RASS Cluster sources"),
    SPIDERS_ERASS_AGN              = (  2, "ERASS AGN sources"),
    SPIDERS_ERASS_CLUS             = (  3, "ERASS Cluster sources"),
    SPIDERS_XMMSL_AGN              = (  4, "XMM Slew survey"),
    SPIDERS_XCLASS_CLUS            = (  5, "XMM serendipitous clusters"),
    TDSS_A                         = ( 20, "Main PanSTARRS selection for TDSS"),
    TDSS_FES_DE                    = ( 21, "TDSS Few epoch spectroscopy"),
    TDSS_FES_DWARFC                = ( 22, "TDSS Few epoch spectroscopy"),
    TDSS_FES_NQHISN                = ( 23, "TDSS Few epoch spectroscopy"),
    TDSS_FES_MGII                  = ( 24, "TDSS Few epoch spectroscopy"),
    TDSS_FES_VARBAL                = ( 25, "TDSS Few epoch spectroscopy"),
    TDSS_B                         = ( 26, "Main TDSS SES version B"),
    TDSS_FES_HYPQSO                = ( 27, "TDSS Few epoch spectroscopy"),
    TDSS_FES_HYPSTAR               = ( 28, "TDSS Few epoch spectroscopy"),
    TDSS_FES_WDDM                  = ( 29, "TDSS Few epoch spectroscopy"),
    TDSS_FES_ACTSTAR               = ( 30, "TDSS Few epoch spectroscopy"),
    TDSS_CP                        = ( 31, "TDSS in common with CORE/PTF"),
    ELG_SCUSS_TEST1                = ( 40, "SCUSS selection for test1 ELG plates"),
    ELG_DES_TEST1                  = ( 41, "DES selection for test1 ELG plates"),
    ELG_DESI_TEST1                 = ( 42, "DESI selection for test1 ELG plates"),
    ELG_SDSS_TEST1                 = ( 43, "SDSS-only selection for test1 ELG plates"),
    ELG_UGRIZW_TEST1               = ( 44, "WISE selection for test1 ELG plates"),
    ELG_UGRIZWbright_TEST1         = ( 45, "WISE selection for test1 ELG plates"),
    ELG_GRIW_TEST1                 = ( 46, "WISE selection for test1 ELG plates"),
)
SEGUE1_TARGET = define_bitmask(
    "SEGUE1_TARGET",
    "SEGUE-1 primary target bits",
    SEGUE1_FG                      = (  9, "F and G stars, based on g-r color (0.2<g-r<0.48 and 14<g<20.2)"),
    SEG1LOW_KG                     = ( 10, "low latitude selection of K-giant stars"),
    SEG1LOW_TO                     = ( 11, "low latitude selection of bluetip stars"),
    SEGUE1_MSWD                    = ( 12, "main-sequence, white dwarf pair"),
    SEGUE1_BHB                     = ( 13, "blue horizontal branch star"),
    SEGUE1_KG                      = ( 14, "K-giants (l and red)"),
    SEGUE1_KD                      = ( 15, "K-dwarfs"),
    SEGUE1_LM                      = ( 16, "low metallicity star"),
    SEGUE1_CWD                     = ( 17, "cool white dwarf"),
    SEGUE1_GD                      = ( 18, "G-dwarf"),
    SEGUE1_WD                      = ( 19, "white dwarf"),
    SEGUE1_MPMSTO                  = ( 20, "metal-poor main sequence turn-off"),
    SEGUE1_BD                      = ( 21, "brown dwarfs"),
    SEGUE1_SDM                     = ( 22, "M sub-dwarfs"),
    SEGUE1_AGB                     = ( 23, "asympototic giant branch stars"),
    SEGUE1_MAN                     = ( 24, "manual selection"),
    SEG1LOW_AGB                    = ( 27, "low latitude selection of AGB stars"),
    SEGUE1_CHECKED                 = ( 31, "was a checked object"),
)
SEGUE1_TARGET2 = define_bitmask(
    "SEGUE1_TARGET2",
    "SEGUE-1 secondary target bits",
    REDDEN_STD                     = (  1, "reddening standard star"),
    SEGUE1_QA                      = (  3, "QA Duplicate Observations (unused)"),
    SKY                            = (  4, "sky target"),
    SPECTROPHOTO_STD               = (  5, "spectrophotometry standard (typically an F-star)"),
    SEGUE1_SCIENCE                 = ( 30, "SEGUE-1 science target"),
    SEGUE1_TEST                    = ( 31, "SEGUE-1 test target"),
)
SEGUE2_TARGET1 = define_bitmask(
    "SEGUE2_TARGET1",
    "SEGUE-2 primary target bits",
    SEGUE2_MSTO                    = (  0, "Main-sequence turnoff"),
    SEGUE2_REDKG                   = (  1, "Red K-giant stars"),
    SEGUE2_LKG                     = (  2, "K-giant star identified by l-color"),
    SEGUE2_PMKG                    = (  3, "K-giant star identified by proper motions"),
    SEGUE2_LM                      = (  4, "Low metallicity"),
    SEGUE2_HVS                     = (  5, "hyper velocity candidate"),
    SEGUE2_XDM                     = (  6, "extreme sdM star"),
    SEGUE2_MII                     = (  7, "M giant"),
    SEGUE2_HHV                     = (  8, "High-velocity halo star candidate"),
    SEGUE2_BHB                     = ( 13, "Blue horizontal branch star"),
    SEGUE2_CWD                     = ( 17, "Cool white dwarf"),
    SEGUE2_CHECKED                 = ( 31, "was a checked object"),
)
SEGUE2_TARGET2 = define_bitmask(
    "SEGUE2_TARGET2",
    "SEGUE-2 secondary target bits",
    LIGHT_TRAP                     = (  0, "light trap hole"),
    SEGUE2_REDDENING               = (  1, "reddening standard"),
    SEGUE2_TEST                    = (  2, "test target"),
    SEGUE2_QA                      = (  3, "repeat target across plates"),
    SKY                            = (  4, "empty area for sky-subtraction"),
    SEGUE2_SPECPHOTO               = (  5, "spectrophotometric star"),
    GUIDE_STAR                     = (  6, "guide star"),
    BUNDLE_HOLE                    = (  7, "bundle hole"),
    QUALITY_HOLE                   = (  8, "quality hole"),
    HOT_STD                        = (  9, "hot standard"),
    SEGUE2_CLUSTER                 = ( 10, "SEGUE-2 stellar cluster target"),
    SEGUE2_STETSON                 = ( 11, "Stetson standard target"),
    SEGUE2_CHECKED                 = ( 31, "was a checked object"),
)
SPECIAL_TARGET1 = define_bitmask(
    "SPECIAL_TARGET1",
    "SDSS special program target bits",
    APBIAS                         = (  0, "aperture bias target"),
    LOWZ_ANNIS                     = (  1, "low-redshift cluster galaxy"),
    QSO_M31                        = (  2, "QSO in M31"),
    COMMISSIONING_STAR             = (  3, "star in commissioning "),
    DISKSTAR                       = (  4, "thin/thick disk star "),
    FSTAR                          = (  5, "F-stars"),
    HYADES_MSTAR                   = (  6, "M-star in Hyades"),
    LOWZ_GALAXY                    = (  7, "low-redshift galaxy"),
    DEEP_GALAXY_RED                = (  8, "deep LRG"),
    DEEP_GALAXY_RED_II             = (  9, "deep LRG"),
    BCG                            = ( 10, "brightest cluster galaxy"),
    MSTURNOFF                      = ( 11, "main sequence turnoff"),
    ORION_BD                       = ( 12, "Brown dwarf in Orion"),
    ORION_MSTAR_EARLY              = ( 13, "Early-type M-star (M0-3) in Orion"),
    ORION_MSTAR_LATE               = ( 14, "Late-type M-star (M4-) in Orion"),
    SPECIAL_FILLER                 = ( 15, "filler from completeTile, check primtarget for details"),
    PHOTOZ_GALAXY                  = ( 16, "test galaxy for photometric redshifts"),
    PREBOSS_QSO                    = ( 17, "QSO for pre-BOSS observations"),
    PREBOSS_LRG                    = ( 18, "QSO for pre-BOSS observations"),
    PREMARVELS                     = ( 19, "pre-MARVELS stellar target"),
    SOUTHERN_EXTENDED              = ( 20, "simple extension of southern targets"),
    SOUTHERN_COMPLETE              = ( 21, "completion in south of main targets"),
    U_PRIORITY                     = ( 22, "priority u-band target "),
    U_EXTRA                        = ( 23, "extra u-band target "),
    U_EXTRA2                       = ( 24, "extra u-band target "),
    FAINT_LRG                      = ( 25, "faint LRG in south"),
    FAINT_QSO                      = ( 26, "faint QSO in south"),
    BENT_RADIO                     = ( 27, "bent double-lobed radio source"),
    STRAIGHT_RADIO                 = ( 28, "straight double-lobed radio source"),
    VARIABLE_HIPRI                 = ( 29, "high priority variable"),
    VARIABLE_LOPRI                 = ( 30, "low priority variable"),
    ALLPSF                         = ( 31, "i<19.1 point sources"),
    ALLPSF_NONSTELLAR              = ( 32, "i<19.1 point sources off stellar locus"),
    ALLPSF_STELLAR                 = ( 33, "i<19.1 point sources on stellar locus"),
    HIPM                           = ( 34, "high proper motion"),
    TAURUS_STAR                    = ( 35, "star on taurus or reddening plate"),
    TAURUS_GALAXY                  = ( 36, "galaxy on taurus or reddening plate"),
    PERSEUS                        = ( 37, "galaxy in perseus-pisces"),
    LOWZ_LOVEDAY                   = ( 38, "low redshift galaxy selected by Loveday "),
)
APOGEE_TARGET1 = define_bitmask(
    "APOGEE_TARGET1",
    "APOGEE primary target bits",
    APOGEE_FAINT                   = (  0, "Selected in faint bin of cohort"),
    APOGEE_MEDIUM                  = (  1, "Selected in medium bin of cohort"),
    APOGEE_BRIGHT                  = (  2, "Selected in bright bin of cohort"),
    APOGEE_IRAC_DERED              = (  3, "Selected using RJCE-IRAC dereddening"),
    APOGEE_WISE_DERED              = (  4, "Selected using RJCE-WISE dereddening"),
    APOGEE_SFD_DERED               = (  5, "Selected using SFD E(B-V) dereddening"),
    APOGEE_NO_DERED                = (  6, "Selected using no dereddening"),
    APOGEE_WASH_GIANT              = (  7, "Selected as giant in Washington photometry"),
    APOGEE_WASH_DWARF              = (  8, "Selected as dwarf in Washington photometry"),
    APOGEE_SCI_CLUSTER             = (  9, "Probable cluster member"),
    APOGEE_EXTENDED                = ( 10, "Extended object"),
    APOGEE_SHORT                   = ( 11, "Short cohort target"),
    APOGEE_INTERMEDIATE            = ( 12, "Intermediate cohort target"),
    APOGEE_LONG                    = ( 13, "Long cohort target"),
    APOGEE_DO_NOT_OBSERVE          = ( 14, "Do not observe (again)"),
    APOGEE_SERENDIPITOUS           = ( 15, "Serendipitously interesting target to reobserve"),
    APOGEE_FIRST_LIGHT             = ( 16, "First list plate target"),
    APOGEE_ANCILLARY               = ( 17, "An ancillary program"),
    APOGEE_M31_CLUSTER             = ( 18, "M31 cluster target (ancillary)"),
    APOGEE_MDWARF                  = ( 19, "M dwarfs selected for RV program (ancillary)"),
    APOGEE_HIRES                   = ( 20, "Star with optical hi-res spectra (ancillary)"),
    APOGEE_OLD_STAR                = ( 21, "Selected as old star (ancillary)"),
    APOGEE_DISK_RED_GIANT          = ( 22, "Disk red giant (ancillary)"),
    APOGEE_KEPLER_EB               = ( 23, "Eclipsing binary from Kepler (ancillary)"),
    APOGEE_GC_PAL1                 = ( 24, "Star in globular cluster (ancillary)"),
    APOGEE_MASSIVE_STAR            = ( 25, "Selected as massive star (ancillary)"),
    APOGEE_SGR_DSPH                = ( 26, "Sagittarius dwarf spheroidal member"),
    APOGEE_KEPLER_SEISMO           = ( 27, "Kepler asteroseismology program target"),
    APOGEE_KEPLER_HOST             = ( 28, "Kepler planet-host program target"),
    APOGEE_FAINT_EXTRA             = ( 29, "Selected as faint target for low target-density field"),
    APOGEE_SEGUE_OVERLAP           = ( 30, "Selected because of overlap with SEGUE survey"),
    APOGEE_CHECKED                 = ( 31, "This target has been checked"),
)
APOGEE_TARGET2 = define_bitmask(
    "APOGEE_TARGET2",
    "APOGEE secondary target bits",
    LIGHT_TRAP                     = (  0, "Light trap"),
    APOGEE_FLUX_STANDARD           = (  1, "Flux standard"),
    APOGEE_STANDARD_STAR           = (  2, "Stellar abundance, parameters standard"),
    APOGEE_RV_STANDARD             = (  3, "Radial velocity standard"),
    SKY                            = (  4, "Sky"),
    SKY_BAD                        = (  5, "Selected as sky but identified as bad (via visual exam or observation)"),
    GUIDE_STAR                     = (  6, "Guide star"),
    BUNDLE_HOLE                    = (  7, "Bundle hole"),
    APOGEE_TELLURIC_BAD            = (  8, "Selected as telluric standard but identified as bad (via SIMBAD or observation)"),
    APOGEE_TELLURIC                = (  9, "Hot (telluric) standard"),
    APOGEE_CALIB_CLUSTER           = ( 10, "Known calibration cluster member"),
    APOGEE_GC_GIANT                = ( 11, "Probable giant in Galactic Center"),
    APOGEE_GC_SUPER_GIANT          = ( 12, "Probable supergiant in Galactic Center"),
    APOGEE_EMBEDDEDCLUSTER_STAR    = ( 13, "Young embedded clusters (ancillary)"),
    APOGEE_LONGBAR                 = ( 14, "Probable RC star in long bar (ancillary)"),
    APOGEE_EMISSION_STAR           = ( 15, "Emission-line star (ancillary)"),
    APOGEE_KEPLER_COOLDWARF        = ( 16, "Kepler cool dwarf/subgiant (ancillary)"),
    APOGEE_MIRCLUSTER_STAR         = ( 17, "Candidate MIR-detected cluster member (ancillary)"),
    APOGEE_CHECKED                 = ( 31, "This target has been checked"),
)
APOGEE_EXTRATARG = define_bitmask(
    "APOGEE_EXTRATARG",
    "APOGEE pixel level mask bits",
    NOT_MAIN                       = (  0, "Not main survey target"),
    COMMISSIONING                  = (  1, "Commissioning data"),
    TELLURIC                       = (  2, "Telluric target"),
    APO1M                          = (  3, "APO1M + APOGEE observation"),
    DUPLICATE                      = (  4, "Duplicate observation of star"),
)
APOGEE_PIXMASK = define_bitmask(
    "APOGEE_PIXMASK",
    "APOGEE extra targeting bits",
    BADPIX                         = (  0, "Pixel marked as BAD in bad pixel mask"),
    CRPIX                          = (  1, "Pixel marked as cosmic ray in ap3d"),
    SATPIX                         = (  2, "Pixel marked as saturated in ap3d"),
    UNFIXABLE                      = (  3, "Pixel marked as unfixable in ap3d"),
    BADDARK                        = (  4, "Pixel marked as bad as determined from dark frame"),
    BADFLAT                        = (  5, "Pixel marked as bad as determined from flat frame"),
    BADERR                         = (  6, "Pixel set to have very high error (not used)"),
    NOSKY                          = (  7, "No sky available for this pixel from sky fibers"),
    LITTROW_GHOST                  = (  8, "Pixel falls in Littrow ghost, may be affected"),
    PERSIST_HIGH                   = (  9, "Pixel falls in high persistence region, may be affected"),
    PERSIST_MED                    = ( 10, "Pixel falls in medium persistence region, may be affected"),
    PERSIST_LOW                    = ( 11, "Pixel falls in low persistence region, may be affected"),
    SIG_SKYLINE                    = ( 12, "Pixel falls near sky line that has significant flux compared with object"),
    SIG_TELLURIC                   = ( 13, "Pixel falls near telluric line that has significant absorption"),
)
APOGEE_STARFLAG = define_bitmask(
    "APOGEE_STARFLAG",
    "APOGEE star-level mask bits",
    BAD_PIXELS                     = (  0, "Spectrum has many bad pixels (>40%): BAD"),
    COMMISSIONING                  = (  1, "Commissioning data (MJD<55761), non-standard configuration, poor LSF: WARN"),
    BRIGHT_NEIGHBOR                = (  2, "Star has neighbor more than 10 times brighter: WARN"),
    VERY_BRIGHT_NEIGHBOR           = (  3, "Star has neighbor more than 100 times brighter: BAD"),
    LOW_SNR                        = (  4, "Spectrum has low S/N (S/N<5): BAD"),
    PERSIST_HIGH                   = (  9, "Spectrum has significant number (>20%) of pixels in high persistence region: WARN"),
    PERSIST_MED                    = ( 10, "Spectrum has significant number (>20%) of pixels in medium persistence region: WARN"),
    PERSIST_LOW                    = ( 11, "Spectrum has significant number (>20%) of pixels in low persistence region: WARN"),
    PERSIST_JUMP_POS               = ( 12, "Spectrum show obvious positive jump in blue chip: WARN"),
    PERSIST_JUMP_NEG               = ( 13, "Spectrum show obvious negative jump in blue chip: WARN"),
    SUSPECT_RV_COMBINATION         = ( 16, "WARNING: RVs from synthetic template differ significantly from those from combined template"),
    SUSPECT_BROAD_LINES            = ( 17, "WARNING: cross-correlation peak with template significantly broader than autocorrelation of template"),
)
APOGEE_ASPCAPFLAG = define_bitmask(
    "APOGEE_ASPCAPFLAG",
    "APOGEE ASPCAP mask bits",
    TEFF_WARN                      = (  0, "WARNING on effective temperature (see PARAMFLAG[0] for details)"),
    LOGG_WARN                      = (  1, "WARNING on log g (see PARAMFLAG[1] for details)"),
    VMICRO_WARN                    = (  2, "WARNING on vmicro (see PARAMFLAG[2] for details)"),
    METALS_WARN                    = (  3, "WARNING on metals (see PARAMFLAG[3] for details)"),
    ALPHAFE_WARN                   = (  4, "WARNING on [alpha/Fe] (see PARAMFLAG[4] for details)"),
    CFE_WARN                       = (  5, "WARNING on [C/Fe] (see PARAMFLAG[5] for details)"),
    NFE_WARN                       = (  6, "WARNING on [N/Fe] (see PARAMFLAG[6] for details)"),
    STAR_WARN                      = (  7, "WARNING overall for star: set if any of TEFF, LOGG, CHI2, COLORTE, ROTATION, SN warn are set"),
    CHI2_WARN                      = (  8, "high chi^2 (> 2*median at ASPCAP temperature (WARN)"),
    COLORTE_WARN                   = (  9, "effective temperature more than 500K from photometric temperature for dereddened color (WARN)"),
    ROTATION_WARN                  = ( 10, "Spectrum has broad lines, with possible bad effects: FWHM of cross-correlation of spectrum with best RV template relative to auto-correltion of template > 1.5 (WARN)"),
    SN_WARN                        = ( 11, "S/N<70 (WARN)"),
    TEFF_BAD                       = ( 16, "BAD effective temperature (see PARAMFLAG[0] for details)"),
    LOGG_BAD                       = ( 17, "BAD log g (see PARAMFLAG[1] for details)"),
    VMICRO_BAD                     = ( 18, "BAD vmicro (see PARAMFLAG[2] for details)"),
    METALS_BAD                     = ( 19, "BAD metals (see PARAMFLAG[3] for details)"),
    ALPHAFE_BAD                    = ( 20, "BAD [alpha/Fe] (see PARAMFLAG[4] for details)"),
    CFE_BAD                        = ( 21, "BAD [C/Fe] (see PARAMFLAG[5] for details)"),
    NFE_BAD                        = ( 22, "BAD [N/Fe] (see PARAMFLAG[6] for details)"),
    STAR_BAD                       = ( 23, "BAD overall for star: set if any of TEFF, LOGG, CHI2, COLORTE, ROTATION, SN error are set, or any parameter is near grid edge (GRIDEDGE_BAD is set in any PARAMFLAG)"),
    CHI2_BAD                       = ( 24, "high chi^2 (> 5*median at ASPCAP temperature (BAD)"),
    COLORTE_BAD                    = ( 25, "effective temperature more than 1000K from photometric temperature for dereddened color (BAD)"),
    ROTATION_BAD                   = ( 26, "Spectrum has broad lines, with possible bad effects: FWHM of cross-correlation of spectrum with best RV template relative to auto-correltion of template > 2 (BAD)"),
    SN_BAD                         = ( 27, "S/N<50 (BAD)"),
    NO_ASPCAP_RESULT               = ( 31, "No result"),
)
APOGEE_PARAMFLAG = define_bitmask(
    "APOGEE_PARAMFLAG",
    "APOGEE parameter mask bits (set for each stellar parameter in ASPCAP fit)",
    GRIDEDGE_BAD                   = (  0, "Parameter within 1/8 grid spacing of grid edge"),
    CALRANGE_BAD                   = (  1, "Parameter outside valid range of calibration determination"),
    OTHER_BAD                      = (  2, "Other error condition"),
    GRIDEDGE_WARN                  = (  8, "Parameter within 1/2 grid spacing of grid edge"),
    CALRANGE_WARN                  = (  9, "Parameter in possibly unreliable range of calibration determination"),
    OTHER_WARN                     = ( 10, "Other warning condition"),
    PARAM_FIXED                    = ( 16, "Parameter set at fixed value, not fit"),
)
APOGEE2_TARGET1 = define_bitmask(
    "APOGEE2_TARGET1",
    "APOGEE2 primary target bits",
    APOGEE2_ONEBIN_GT_0_5          = (  0, "Selected in single (J-Ks)o > 0.5 color bin"),
    APOGEE2_TWOBIN_0_5_TO_0_8      = (  1, "Selected in blue 0.5 < (J-Ks)o < 0.8 color bin"),
    APOGEE2_TWOBIN_GT_0_8          = (  2, "Selected in red (J-Ks)o > 0.8 color bin"),
    APOGEE2_IRAC_DERED             = (  3, "Selected with RJCE-IRAC dereddening"),
    APOGEE2_WISE_DERED             = (  4, "Selected with RJCE-WISE dereddening"),
    APOGEE2_SFD_DERED              = (  5, "Selected with SFD_EBV dereddening"),
    APOGEE2_NO_DERED               = (  6, "Selected with no dereddening"),
    APOGEE2_WASH_GIANT             = (  7, "Selected as Wash+DDO51 photometric giant"),
    APOGEE2_WASH_DWARF             = (  8, "Selected as Wash+DDO51 photometric dwarf"),
    APOGEE2_SCI_CLUSTER            = (  9, "Science cluster candidate member"),
    APOGEE2_SHORT                  = ( 11, "Selected as part of a short cohort"),
    APOGEE2_MEDIUM                 = ( 12, "Selected as part of a medium cohort"),
    APOGEE2_LONG                   = ( 13, "Selected as part of a long cohort"),
    APOGEE2_NORMAL_SAMPLE          = ( 14, "Selected as part of the random sample"),
    APOGEE2_MANGA_LED              = ( 15, "Star on a shared MaNGA-led design"),
    APOGEE2_ONEBIN_GT_0_3          = ( 16, "Selected in single (J-Ks)o > 0.3 color bin"),
    APOGEE2_WASH_NOCLASS           = ( 17, "Selected because it has no W+D classification"),
    APOGEE2_STREAM_MEMBER          = ( 18, "Selected as confirmed halo tidal stream member"),
    APOGEE2_STREAM_CANDIDATE       = ( 19, "Selected as potential halo tidal stream member (based on photometry)"),
    APOGEE2_DSPH_MEMBER            = ( 20, "Selected as confirmed dSph member (non Sgr)"),
    APOGEE2_DSPH_CANDIDATE         = ( 21, "Selected as potential dSph member (non Sgr) (based on photometry)"),
    APOGEE2_MAGCLOUD_MEMBER        = ( 22, "Selected as confirmed Mag Cloud member"),
    APOGEE2_MAGCLOUD_CANDIDATE     = ( 23, "Selected as potential Mag Cloud member (based on photometry)"),
    APOGEE2_RRLYR                  = ( 24, "Selected as a bulge RR Lyrae star"),
    APOGEE2_SGR_DSPH               = ( 26, "Selected as confirmed Sgr core/stream member"),
    APOGEE2_APOKASC_GIANT          = ( 27, "Selected as part of APOKASC giant sample"),
    APOGEE2_APOKASC_DWARF          = ( 28, "Selected as part of APOKASC dwarf sample"),
    APOGEE2_FAINT_EXTRA            = ( 29, "Faint star (fainter than cohort limit; not required to reach survey S/N requirement)"),
    APOGEE2_APOKASC                = ( 30, "Selected as part of the APOKASC program (incl. seismic/gyro targets and others)"),
)
APOGEE2_TARGET2 = define_bitmask(
    "APOGEE2_TARGET2",
    "APOGEE2 secondary target bits",
    APOGEE2_STANDARD_STAR          = (  2, "Stellar parameters/abundance standard"),
    APOGEE2_RV_STANDARD            = (  3, "Stellar RV standard"),
    APOGEE2_SKY                    = (  4, "Sky fiber"),
    APOGEE2_EXTERNAL_CALIB         = (  5, "External survey calibration target (generic flag; others below dedicated to specific surveys)"),
    APOGEE2_INTERNAL_CALIB         = (  6, "Internal survey calibration target (observed in at least 2 of: APOGEE-1, -2N, -2S)"),
    APOGEE2_TELLURIC               = (  9, "Telluric calibrator target"),
    APOGEE2_CALIB_CLUSTER          = ( 10, "Selected as calibration cluster member"),
    APOGEE2_LITERATURE_CALIB       = ( 13, "Overlap with high-resolution literature studies"),
    APOGEE2_GES_OVERLAP            = ( 14, "Overlap with Gaia-ESO"),
    APOGEE2_ARGOS_OVERLAP          = ( 15, "Overlap with ARGOS"),
    APOGEE2_GAIA_OVERLAP           = ( 16, "Overlap with Gaia"),
    APOGEE2_GALAH_OVERLAP          = ( 17, "Overlap with GALAH"),
    APOGEE2_RAVE_OVERLAP           = ( 18, "Overlap with RAVE"),
    APOGEE2_1M_TARGET              = ( 22, "Selected as a 1-m target"),
    APOGEE2_OBJECT                 = ( 30, "This object is an APOGEE-2 target"),
)
APOGEE2_TARGET3 = define_bitmask(
    "APOGEE2_TARGET3",
    "APOGEE2 trinary target bits",
    APOGEE2_KOI                    = (  0, "Selected as part of the long cadence KOI study"),
    APOGEE2_EB                     = (  1, "Selected as part of the EB program"),
    APOGEE2_KOI_CONTROL            = (  2, "Selected as part of the long cadence KOI control sample"),
    APOGEE2_MDWARF                 = (  3, "Selected as part of the M dwarf study"),
    APOGEE2_SUBSTELLAR_COMPANIONS  = (  4, "Selected as part of the substellar companion search"),
    APOGEE2_YOUNG_CLUSTER          = (  5, "Selected as part of the young cluster study (IN-SYNC)"),
    APOGEE2_ANCILLARY              = (  8, "Selected as an ancillary target"),
    APOGEE2_MASSIVE_STAR           = (  9, "Selected as part of the Massive Star program"),
)
MANGA_DRP2QUAL = define_bitmask(
    "MANGA_DRP2QUAL",
    "Mask bits for MaNGA DRP-2d quality flags",
    VALIDFILE                      = (  0, "File is valid"),
    EXTRACTBAD                     = (  1, "Many bad values in extracted frame"),
    EXTRACTBRIGHT                  = (  2, "Extracted spectra abnormally bright"),
    LOWEXPTIME                     = (  3, "Exposure time less than 10 minutes"),
    BADIFU                         = (  4, "One or more IFUs missing/bad in this frame"),
    HIGHSCAT                       = (  5, "High scattered light levels"),
    SCATFAIL                       = (  6, "Failure to correct high scattered light levels"),
    BADDITHER                      = (  7, "Bad dither location information"),
    ARCFOCUS                       = (  8, " Bad focus on arc frames"),
    RAMPAGINGBUNNY                 = (  9, " Rampaging dust bunnies in IFU flats"),
    SKYSUBBAD                      = ( 10, "Bad sky subtraction"),
    SKYSUBFAIL                     = ( 11, "Failed sky subtraction"),
    FULLCLOUD                      = ( 12, "Completely cloudy exposure"),
)
MANGA_DRP3QUAL = define_bitmask(
    "MANGA_DRP3QUAL",
    "Mask bits for MaNGA DRP-3d quality flags",
    VALIDFILE                      = (  0, "File is valid"),
    BADDEPTH                       = (  1, "IFU does not reach target depth"),
    SKYSUBBAD                      = (  2, " Bad sky subtraction in one or more frames"),
    HIGHSCAT                       = (  3, " High scattered light in one or more frames"),
    BADASTROM                      = (  4, " Bad astrometry in one or more frames"),
    VARIABLELSF                    = (  5, " LSF varies signif. between component spectra"),
    BADOMEGA                       = (  6, " Omega greater than threshhold in one or more sets"),
    BADSET                         = (  7, " One or more sets are bad"),
    BADFLUX                        = (  8, " Bad flux calibration"),
    CRITICAL                       = ( 30, " Critical failure in one or more frames"),
)
MANGA_DAPQUAL = define_bitmask(
    "MANGA_DAPQUAL",
    "Mask bits for MaNGA DAP quality flags",
    VALIDFILE                      = (  0, "File is valid"),
)
MANGA_DRP2PIXMASK = define_bitmask(
    "MANGA_DRP2PIXMASK",
    "Mask bits per fiber or pixel for 2d MaNGA spectra.",
    NOPLUG                         = (  0, "Fiber not listed in plugmap file"),
    BADTRACE                       = (  1, "Bad trace"),
    BADFLAT                        = (  2, "Low counts in fiberflat"),
    BADARC                         = (  3, "Bad arc solution"),
    MANYBADCOLUMNS                 = (  4, "More than 10% of pixels are bad columns"),
    MANYREJECTED                   = (  5, "More than 10% of pixels are rejected in extraction"),
    LARGESHIFT                     = (  6, "Large spatial shift between flat and object position"),
    BADSKYFIBER                    = (  7, "Sky fiber shows extreme residuals"),
    NEARWHOPPER                    = (  8, "Within 2 fibers of a whopping fiber (exclusive)"),
    WHOPPER                        = (  9, "Whopping fiber, with a very bright source."),
    SMEARIMAGE                     = ( 10, "Smear available for red and blue cameras"),
    SMEARHIGHSN                    = ( 11, "S/N sufficient for full smear fit"),
    SMEARMEDSN                     = ( 12, "S/N only sufficient for scaled median fit"),
    DEADFIBER                      = ( 13, "Broken fiber according to metrology files"),
    BADPIX                         = ( 15, "Pixel flagged in badpix reference file."),
    COSMIC                         = ( 16, "Pixel flagged as cosmic ray."),
    NEARBADPIXEL                   = ( 17, "Bad pixel within 3 pixels of trace."),
    LOWFLAT                        = ( 18, "Flat field less than 0.5"),
    FULLREJECT                     = ( 19, "Pixel fully rejected in extraction model fit (INVVAR=0)"),
    PARTIALREJECT                  = ( 20, "Some pixels rejected in extraction model fit"),
    SCATTEREDLIGHT                 = ( 21, "Scattered light significant"),
    CROSSTALK                      = ( 22, "Cross-talk significant"),
    NOSKY                          = ( 23, "Sky level unknown at this wavelength (INVVAR=0)"),
    BRIGHTSKY                      = ( 24, "Sky level > flux + 10*(flux_err) AND sky > 1.25 * median(sky,99 pixels)"),
    NODATA                         = ( 25, "No data available in combine B-spline (INVVAR=0)"),
    COMBINEREJ                     = ( 26, "Rejected in combine B-spline"),
    BADFLUXFACTOR                  = ( 27, "Low flux-calibration or flux-correction factor"),
    BADSKYCHI                      = ( 28, "Relative chi^2 > 3 in sky residuals at this wavelength"),
    REDMONSTER                     = ( 29, "Contiguous region of bad chi^2 in sky residuals (with threshhold of relative chi^2 > 3)."),
    _3DREJECT                      = ( 30, "Used in RSS file, indicates should be rejected when making 3D cube"),
)
MANGA_DRP3PIXMASK = define_bitmask(
    "MANGA_DRP3PIXMASK",
    "Mask bits per spaxel for a MaNGA data cube.",
    NOCOV                          = (  0, "No coverage in cube"),
    LOWCOV                         = (  1, "Low coverage depth in cube"),
    DEADFIBER                      = (  2, "Major contributing fiber is dead"),
    FORESTAR                       = (  3, "Foreground star"),
    DONOTUSE                       = ( 10, "Do not use this spaxel for science"),
)
MANGA_TARGET1 = define_bitmask(
    "MANGA_TARGET1",
    "Mask bits identifying galaxy samples.",
    NONE                           = (  0, " "),
    PRIMARY_PLUS_COM               = (  1, " March 2014 commissioning"),
    SECONDARY_COM                  = (  2, " March 2014 commissioning"),
    COLOR_ENHANCED_COM             = (  3, " March 2014 commissioning"),
    PRIMARY_v1_1_0                 = (  4, " First tag, August 2014 plates"),
    SECONDARY_v1_1_0               = (  5, " First tag, August 2014 plates"),
    COLOR_ENHANCED_v1_1_0          = (  6, " First tag, August 2014 plates"),
    PRIMARY_COM2                   = (  7, " July 2014 commissioning"),
    SECONDARY_COM2                 = (  8, " July 2014 commissioning"),
    COLOR_ENHANCED_COM2            = (  9, " July 2014 commissioning"),
    PRIMARY_v1_2_0                 = ( 10, " "),
    SECONDARY_v1_2_0               = ( 11, " "),
    COLOR_ENHANCED_v1_2_0          = ( 12, " "),
    FILLER                         = ( 13, " Filler targets"),
    ANCILLARY                      = ( 14, " Ancillary program targets"),
)
MANGA_TARGET2 = define_bitmask(
    "MANGA_TARGET2",
    "Mask bits identifying non-galaxy samples.",
    NONE                           = (  0, " "),
    SKY                            = (  1, " "),
    STELLIB_SDSS_COM               = (  2, "Commissioning selection using SDSS photometry"),
    STELLIB_2MASS_COM              = (  3, "Commissioning selection using 2MASS photometry"),
    STELLIB_KNOWN_COM              = (  4, "Commissioning selection of known parameter stars"),
    STELLIB_COM_mar2015            = (  5, "Commissioning selection in March 2015"),
    STD_FSTAR_COM                  = ( 20, " "),
    STD_WD_COM                     = ( 21, " "),
    STD_STD_COM                    = ( 22, " "),
    STD_FSTAR                      = ( 23, " "),
    STD_WD                         = ( 24, " "),
    STD_APASS_COM                  = ( 25, "Commissioning selection of stds using APASS photometry"),
)
MANGA_TARGET3 = define_bitmask(
    "MANGA_TARGET3",
    "Mask bits identifying ancillary samples.",
    NONE                           = (  0, " "),
    AGN_BAT                        = (  1, " "),
    AGN_OIII                       = (  2, " "),
    AGN_WISE                       = (  3, " "),
    AGN_PALOMAR                    = (  4, " "),
    VOID                           = (  5, " "),
    EDGE_ON_WINDS                  = (  6, " "),
    PAIR_ENLARGE                   = (  7, " "),
    PAIR_RECENTER                  = (  8, " "),
    PAIR_SIM                       = (  9, " "),
    PAIR_2IFU                      = ( 10, " "),
    LETTERS                        = ( 11, " "),
    MASSIVE                        = ( 12, " "),
    MWA                            = ( 13, " "),
    DWARF                          = ( 14, " "),
    RADIO_JETS                     = ( 15, " "),
    DISKMASS                       = ( 16, " "),
    BCG                            = ( 17, " "),
    ANGST                          = ( 18, " "),
    DEEP_COMA                      = ( 19, " "),
)

if __name__ == '__main__':
   print(decode_bitmask(SPPIXMASK, 33554432)) 
