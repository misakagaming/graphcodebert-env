namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: CYYY0321_TYPE_READ               Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:40:13
  //    Target DBMS: ODBC/ADO.NET              User: AliAl
  //    Access Method: <NONE>         
  //
  //    Generation options:
  //    Debug trace option not selected
  //    Data modeling constraint enforcement not selected
  //    Optimized import view initialization not selected
  //    Enforce default values with DBMS not selected
  //    Init unspecified optional fields to NULL not selected
  //
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // using Statements
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  using System;
  using com.ca.gen.vwrt;
  using com.ca.gen.vwrt.types;
  using com.ca.gen.vwrt.vdf;
  using com.ca.gen.csu.exception;
  
  using com.ca.gen.abrt;
  using com.ca.gen.abrt.functions;
  using com.ca.gen.abrt.cascade;
  using com.ca.gen.abrt.manager;
  using com.ca.gen.abrt.trace;
  using com.ca.gen.exits.common;
  using com.ca.gen.odc;
  using System.Data;
  using System.Collections;
  
  public class CYYY0321 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // IMPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    CYYY0321_IA WIa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    CYYY0321_OA WOa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // LOCAL VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    CYYY0321_LA WLa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.DYYY0321_IA Dyyy0321Ia;
    GEN.ORT.YYY.DYYY0321_OA Dyyy0321Oa;
    GEN.ORT.YYY.CYYY9001_OA Cyyy9001Oa;
    GEN.ORT.YYY.CYYY9141_IA Cyyy9141Ia;
    GEN.ORT.YYY.CYYY9141_OA Cyyy9141Oa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // REPEATING GROUP VIEW STATUS FIELDS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool LocGroupContext_FL_001;
    int LocGroupContext_PS_001;
    bool LocGroupContext_RF_001;
    public const int LocGroupContext_MM_001 = 9;
    bool ImpGroupContext_FL_002;
    int ImpGroupContext_PS_002;
    bool ImpGroupContext_RF_002;
    public const int ImpGroupContext_MM_002 = 9;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020147_esc_flag;
    bool func_0020972742_esc_flag;
    bool func_0020972260_esc_flag;
    bool func_0020972166_esc_flag;
    //       +->   CYYY0321_TYPE_READ                01/09/2024  13:40
    //       !       IMPORTS:
    //       !         Entity View imp type (Transient, Mandatory, Import
    //       !                     only)
    //       !           tinstance_id
    //       !           tkey_attr_text
    //       !       EXPORTS:
    //       !         Entity View exp type (Transient, Export only)
    //       !           tinstance_id
    //       !           treference_id
    //       !           tcreate_user_id
    //       !           tupdate_user_id
    //       !           tkey_attr_text
    //       !           tsearch_attr_text
    //       !           tother_attr_text
    //       !           tother_attr_date
    //       !           tother_attr_time
    //       !           tother_attr_amount
    //       !         Work View exp_error iyy1_component (Transient, Export
    //       !                     only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !       LOCALS:
    //       !         Work View loc_error iyy1_component
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !         Group View (9) loc_group_context
    //       !           Work View loc_g_context dont_change_text
    //       !             text_150
    //       !         Work View loc dont_change_return_codes
    //       !           1_ok
    //       !         Work View loc dont_change_reason_codes
    //       !           1_default
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !   
    //     2 !   
    //     3 !   
    //     4 !  NOTE: 
    //     4 !  Please review explanation for purpose.
    //     4 !  
    //     5 !  NOTE: 
    //     5 !  RELEASE HISTORY
    //     5 !  01_00 23-02-1998 New release
    //     5 !  
    //     6 !  USE cyyy9001_exception_hndlng_dflt
    //     6 !     WHICH EXPORTS: Work View exp_error iyy1_component FROM Work
    //     6 !              View exp_error iyy1_component
    //     7 !   
    //     8 !  NOTE: 
    //     8 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     8 !  Please rename the procedure name below.
    //     8 !  
    //     9 !  SET SUBSCRIPT OF loc_group_context TO 1 
    //    10 !  SET loc_g_context dont_change_text text_150 TO "CYYY0321" 
    //    11 !  SET SUBSCRIPT OF loc_group_context TO 2 
    //    12 !  SET loc_g_context dont_change_text text_150 TO "READ" 
    //    13 !   
    //    14 !  NOTE: 
    //    14 !  ****************************************************************
    //    14 !  Values of the ReturnCode/ReasonCode used.
    //    14 !  
    //    15 !  NOTE: 
    //    15 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    15 !  Please set the return ve reason code values below.
    //    15 !  
    //    16 !  SET loc dont_change_return_codes 1_ok TO 1 
    //    17 !   
    //    18 !  SET loc dont_change_reason_codes 1_default TO 1 
    //    19 !   
    //    20 !  +->IF exp_error iyy1_component return_code >= loc
    //    20 !  !        dont_change_return_codes 1_ok
    //    21 !  !  USE dyyy0321_type_read
    //    21 !  !     WHICH IMPORTS: Work View exp_error iyy1_component TO
    //    21 !  !              Work View imp_error iyy1_component
    //    21 !  !                    Entity View imp type TO Entity View imp
    //    21 !  !              type
    //    21 !  !     WHICH EXPORTS: Entity View exp type FROM Entity View exp
    //    21 !  !              type
    //    21 !  !                    Work View loc_error iyy1_component FROM
    //    21 !  !              Work View exp_error iyy1_component
    //    22 !  !   
    //    23 !  !  NOTE: 
    //    23 ...**************************************************************
    //    23 ...**
    //    23 ...If External will be USEd the code sample replacement for
    //    23 ...above code:
    //    23 ...
    //    23 ...| USE eyyy0321_type_read
    //    23 ...|    WHICH IMPORTS: Entity View imp type  TO Entity View imp
    //    23 ...type 
    //    23 ...|    WHICH EXPORTS: Entity View exp type  FROM Entity View
    //    23 ...exp type 
    //    23 ...|                     Work View   loc_error d._c._text  FROM
    //    23 ...Work View   exp_error d._c._text 
    //    23 ...|
    //    23 ...| +- CASE OF loc_error dont_change_text text_2 
    //    23 ...| +- CASE "OK" 
    //    23 ...| +- CASE "NF" 
    //    23 ...| |  SET exp_error iyy1_com.. return_code TO loc
    //    23 ...d._c._return_codes n10_obj_not_found 
    //    23 ...| |  SET exp_error iyy1_com.. reason_code TO loc
    //    23 ...d._c._reason_codes 141_type_not_found 
    //    23 ...| +- OTHERWISE 
    //    23 ...| |  SET exp_error iyy1_com.. return_code TO loc
    //    23 ...d._c._return_codes n999_unexpected_exception 
    //    23 ...| |  SET exp_error iyy1_com.. reason_code TO loc
    //    23 ...d._c._reason_codes 1_default 
    //    23 ...| +--
    //    23 ...
    //    24 !  !  +->IF loc_error iyy1_component return_code < loc
    //    24 !  !  !        dont_change_return_codes 1_ok
    //    25 !  !  !  MOVE loc_error iyy1_component TO exp_error
    //    25 !  !  !              iyy1_component
    //    26 !  !  !  NOTE: 
    //    26 ...!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    26 ...!!!!!!!!!!!
    //    26 ...!! ATTENTION : In D AB, if context string must be formed
    //    26 ...ESCAPE AB must be exited.
    //    26 ...
    //    27 ! <------ESCAPE
    //    24 !  !  +--
    //    20 !  +--
    //    28 !   
    //    29 !  +->IF exp_error iyy1_component return_code < loc
    //    29 !  !        dont_change_return_codes 1_ok
    //    30 !  !  USE cyyy9141_context_string_set
    //    30 !  !     WHICH IMPORTS: Group View  loc_group_context TO Group
    //    30 !  !              View imp_group_context
    //    30 !  !     WHICH EXPORTS: Work View loc_error iyy1_component FROM
    //    30 !  !              Work View exp_error iyy1_component
    //    30 !  !                    Work View exp_error iyy1_component FROM
    //    30 !  !              Work View exp_context iyy1_component
    //    31 !  !   
    //    32 !  !  +->IF loc_error iyy1_component return_code < loc
    //    32 !  !  !        dont_change_return_codes 1_ok
    //    33 !  !  !  MOVE loc_error iyy1_component TO exp_error
    //    33 !  !  !              iyy1_component
    //    32 !  !  +--
    //    34 !  !  SET exp_error iyy1_component severity_code TO "E" 
    //    29 !  +--
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public CYYY0321(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:40:13";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "CYYY0321_TYPE_READ";
      NestingLevel = 0;
      ValChkDeadlockTimeout = false;
      ValChkDBError = false;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK FUNCTION DECLARATIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void Execute( Object in_runtime_parm1, 
    	IRuntimePStepContext in_runtime_parm2, 
    	GlobData in_globdata, 
    	CYYY0321_IA import_view, 
    	CYYY0321_OA export_view )
    {
      IefRuntimeParm1 = in_runtime_parm1;
      IefRuntimeParm2 = in_runtime_parm2;
      Globdata = in_globdata;
      WIa = import_view;
      WOa = export_view;
      _Execute();
    }
    
    private void _Execute()
    {
      
      f_22020147_localAlloc( "CYYY0321_TYPE_READ" );
      if ( Globdata.GetErrorData().GetLastStatus() == ErrorData.LastStatusIefAllocationError )
      	return;
      
      ++(NestingLevel);
      try {
        f_22020147_init(  );
        f_22020147(  );
      } catch( Exception e ) {
        if ( ((Globdata.GetErrorData().GetStatus() == ErrorData.StatusNone) && (Globdata.GetErrorData().GetErrorEncountered() == 
          ErrorData.ErrorEncounteredNoErrorFound)) && (Globdata.GetErrorData().GetViewOverflow() == 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
          Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusUnexpectedExceptionError );
          Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
          Globdata.GetErrorData(  ).SetErrorMessage( e );
        }
      }
      --(NestingLevel);
    }
    public void f_22020147(  )
    {
      func_0022020147_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020147" );
      Globdata.GetStateData().SetCurrentABName( "CYYY0321_TYPE_READ" );
      {
        
        
        
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //    Please review explanation for purpose.                          
        //                                                                    
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //    RELEASE HISTORY                                                 
        //    01_00 23-02-1998 New release                                    
        //                                                                    
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        Globdata.GetStateData().SetLastStatementNumber( "0000000006" );
        
        Cyyy9001Oa = (GEN.ORT.YYY.CYYY9001_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY9001).Assembly,
        	"GEN.ORT.YYY.CYYY9001_OA" ));
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
        IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.CYYY9001).Assembly,
        	"GEN.ORT.YYY.CYYY9001",
        	"Execute",
        	null,
        	Cyyy9001Oa );
        if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
          ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          throw new ABException();
        }
        Globdata.GetStateData().SetCurrentABId( "0022020147" );
        Globdata.GetStateData().SetCurrentABName( "CYYY0321_TYPE_READ" );
        Globdata.GetStateData().SetLastStatementNumber( "0000000006" );
        WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy9001Oa.ExpErrorIyy1ComponentSeverityCode, 1);
        WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Cyyy9001Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
        WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Cyyy9001Oa.ExpErrorIyy1ComponentOriginServid);
        WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Cyyy9001Oa.ExpErrorIyy1ComponentContextString, 512);
        WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Cyyy9001Oa.ExpErrorIyy1ComponentReturnCode);
        WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Cyyy9001Oa.ExpErrorIyy1ComponentReasonCode);
        WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Cyyy9001Oa.ExpErrorIyy1ComponentChecksum, 15);
        Cyyy9001Oa.FreeInstance(  );
        Cyyy9001Oa = null;
        
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                      
        //    Please rename the procedure name below.                         
        //                                                                    
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        Globdata.GetStateData().SetLastStatementNumber( "0000000009" );
        LocGroupContext_PS_001 = (int)TIRD2DEC.Execute1(1, 0, TIRD2DEC.ROUND_NONE, 0);
        if ( (LocGroupContext_PS_001 > WLa.LocGroupContext_MA) && (LocGroupContext_PS_001 <= 9) )
        WLa.LocGroupContext_MA = IntAttr.ValueOf(LocGroupContext_PS_001);
        Globdata.GetStateData().SetLastStatementNumber( "0000000010" );
        WLa.LocGContextDontChangeTextText150[LocGroupContext_PS_001-1] = FixedStringAttr.ValueOf("CYYY0321", 150);
        f_173015126_rgvc(  );
        Globdata.GetStateData().SetLastStatementNumber( "0000000011" );
        LocGroupContext_PS_001 = (int)TIRD2DEC.Execute1(2, 0, TIRD2DEC.ROUND_NONE, 0);
        if ( (LocGroupContext_PS_001 > WLa.LocGroupContext_MA) && (LocGroupContext_PS_001 <= 9) )
        WLa.LocGroupContext_MA = IntAttr.ValueOf(LocGroupContext_PS_001);
        Globdata.GetStateData().SetLastStatementNumber( "0000000012" );
        WLa.LocGContextDontChangeTextText150[LocGroupContext_PS_001-1] = FixedStringAttr.ValueOf("READ", 150);
        f_173015126_rgvc(  );
        
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //    **************************************************************  
        //    **                                                              
        //    Values of the ReturnCode/ReasonCode used.                       
        //                                                                    
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
        //    !!                                                              
        //    Please set the return ve reason code values below.              
        //                                                                    
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        Globdata.GetStateData().SetLastStatementNumber( "0000000016" );
        WLa.LocDontChangeReturnCodesQ1Ok = IntAttr.ValueOf((int)TIRD2DEC.Execute1(1, 0, TIRD2DEC.ROUND_NONE, 5));
        
        Globdata.GetStateData().SetLastStatementNumber( "0000000018" );
        WLa.LocDontChangeReasonCodesQ1Default = IntAttr.ValueOf((int)TIRD2DEC.Execute1(1, 0, TIRD2DEC.ROUND_NONE, 5));
        
        Globdata.GetStateData().SetLastStatementNumber( "0000000020" );
        if ( ((double) WOa.ExpErrorIyy1ComponentReturnCode >= (double) WLa.LocDontChangeReturnCodesQ1Ok) )
        {
          f_20972742(  );
        }
        
        if ( func_0022020147_esc_flag )
        {
          goto f_0022020147_esctag;
        }
        
        Globdata.GetStateData().SetLastStatementNumber( "0000000029" );
        if ( ((double) WOa.ExpErrorIyy1ComponentReturnCode < (double) WLa.LocDontChangeReturnCodesQ1Ok) )
        {
          f_20972260(  );
        }
        
      }
      f_0022020147_esctag: 
      ;
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    
    public void f_20972742(  )
    {
      func_0020972742_esc_flag = false;
      {
        Globdata.GetStateData().SetLastStatementNumber( "0000000021" );
        
        Dyyy0321Ia = (GEN.ORT.YYY.DYYY0321_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.DYYY0321).Assembly,
        	"GEN.ORT.YYY.DYYY0321_IA" ));
        Dyyy0321Oa = (GEN.ORT.YYY.DYYY0321_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.DYYY0321).Assembly,
        	"GEN.ORT.YYY.DYYY0321_OA" ));
        Dyyy0321Ia.ImpTypeTinstanceId = TimestampAttr.ValueOf(WIa.ImpTypeTinstanceId);
        Dyyy0321Ia.ImpTypeTkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpTypeTkeyAttrText, 4);
        Dyyy0321Ia.ImpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(WOa.ExpErrorIyy1ComponentSeverityCode, 1);
        Dyyy0321Ia.ImpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(WOa.ExpErrorIyy1ComponentRollbackIndicator, 1);
        Dyyy0321Ia.ImpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(WOa.ExpErrorIyy1ComponentOriginServid);
        Dyyy0321Ia.ImpErrorIyy1ComponentContextString = StringAttr.ValueOf(WOa.ExpErrorIyy1ComponentContextString, 512);
        Dyyy0321Ia.ImpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(WOa.ExpErrorIyy1ComponentReturnCode);
        Dyyy0321Ia.ImpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(WOa.ExpErrorIyy1ComponentReasonCode);
        Dyyy0321Ia.ImpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(WOa.ExpErrorIyy1ComponentChecksum, 15);
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
        IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.DYYY0321).Assembly,
        	"GEN.ORT.YYY.DYYY0321",
        	"Execute",
        	Dyyy0321Ia,
        	Dyyy0321Oa );
        if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
          ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          throw new ABException();
        }
        Globdata.GetStateData().SetCurrentABId( "0022020147" );
        Globdata.GetStateData().SetCurrentABName( "CYYY0321_TYPE_READ" );
        Globdata.GetStateData().SetLastStatementNumber( "0000000021" );
        WLa.LocErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Dyyy0321Oa.ExpErrorIyy1ComponentSeverityCode, 1);
        WLa.LocErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Dyyy0321Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
        WLa.LocErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Dyyy0321Oa.ExpErrorIyy1ComponentOriginServid);
        WLa.LocErrorIyy1ComponentContextString = StringAttr.ValueOf(Dyyy0321Oa.ExpErrorIyy1ComponentContextString, 512);
        WLa.LocErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Dyyy0321Oa.ExpErrorIyy1ComponentReturnCode);
        WLa.LocErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Dyyy0321Oa.ExpErrorIyy1ComponentReasonCode);
        WLa.LocErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Dyyy0321Oa.ExpErrorIyy1ComponentChecksum, 15);
        WOa.ExpTypeTinstanceId = TimestampAttr.ValueOf(Dyyy0321Oa.ExpTypeTinstanceId);
        WOa.ExpTypeTreferenceId = TimestampAttr.ValueOf(Dyyy0321Oa.ExpTypeTreferenceId);
        WOa.ExpTypeTcreateUserId = FixedStringAttr.ValueOf(Dyyy0321Oa.ExpTypeTcreateUserId, 8);
        WOa.ExpTypeTupdateUserId = FixedStringAttr.ValueOf(Dyyy0321Oa.ExpTypeTupdateUserId, 8);
        WOa.ExpTypeTkeyAttrText = FixedStringAttr.ValueOf(Dyyy0321Oa.ExpTypeTkeyAttrText, 4);
        WOa.ExpTypeTsearchAttrText = FixedStringAttr.ValueOf(Dyyy0321Oa.ExpTypeTsearchAttrText, 20);
        WOa.ExpTypeTotherAttrText = FixedStringAttr.ValueOf(Dyyy0321Oa.ExpTypeTotherAttrText, 2);
        WOa.ExpTypeTotherAttrDate = DateAttr.ValueOf(Dyyy0321Oa.ExpTypeTotherAttrDate);
        WOa.ExpTypeTotherAttrTime = TimeAttr.ValueOf(Dyyy0321Oa.ExpTypeTotherAttrTime);
        WOa.ExpTypeTotherAttrAmount = DecimalAttr.ValueOf(Dyyy0321Oa.ExpTypeTotherAttrAmount);
        Dyyy0321Ia.FreeInstance(  );
        Dyyy0321Ia = null;
        Dyyy0321Oa.FreeInstance(  );
        Dyyy0321Oa = null;
        
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //    **************************************************************  
        //    **                                                              
        //    If External will be USEd the code sample replacement for        
        //    above code:                                                     
        //                                                                    
        //    | USE eyyy0321_type_read                                        
        //    |    WHICH IMPORTS: Entity View imp type  TO Entity View imp    
        //    type                                                            
        //    |    WHICH EXPORTS: Entity View exp type  FROM Entity View      
        //    exp type                                                        
        //    |                     Work View   loc_error d._c._text  FROM    
        //    Work View   exp_error d._c._text                                
        //    |                                                               
        //    | +- CASE OF loc_error dont_change_text text_2                  
        //    | +- CASE "OK"                                                  
        //    | +- CASE "NF"                                                  
        //    | |  SET exp_error iyy1_com.. return_code TO loc                
        //    d._c._return_codes n10_obj_not_found                            
        //    | |  SET exp_error iyy1_com.. reason_code TO loc                
        //    d._c._reason_codes 141_type_not_found                           
        //    | +- OTHERWISE                                                  
        //    | |  SET exp_error iyy1_com.. return_code TO loc                
        //    d._c._return_codes n999_unexpected_exception                    
        //    | |  SET exp_error iyy1_com.. reason_code TO loc                
        //    d._c._reason_codes 1_default                                    
        //    | +--                                                           
        //                                                                    
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        Globdata.GetStateData().SetLastStatementNumber( "0000000024" );
        if ( ((double) WLa.LocErrorIyy1ComponentReturnCode < (double) WLa.LocDontChangeReturnCodesQ1Ok) )
        {
          f_20972166(  );
        }
        
      }
      f_0020972742_esctag: 
      ;
      return;
    }
    
    public void f_20972260(  )
    {
      func_0020972260_esc_flag = false;
      Globdata.GetStateData().SetLastStatementNumber( "0000000030" );
      
      Cyyy9141Ia = (GEN.ORT.YYY.CYYY9141_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY9141).Assembly,
      	"GEN.ORT.YYY.CYYY9141_IA" ));
      Cyyy9141Oa = (GEN.ORT.YYY.CYYY9141_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY9141).Assembly,
      	"GEN.ORT.YYY.CYYY9141_OA" ));
      Cyyy9141Ia.ImpGroupContext_MA = IntAttr.ValueOf(WLa.LocGroupContext_MA);
      for(Adim1 = 1; Adim1 <= WLa.LocGroupContext_MA; ++(Adim1))
      {
        Cyyy9141Ia.ImpGContextDontChangeTextText150[Adim1-1] = FixedStringAttr.ValueOf(WLa.LocGContextDontChangeTextText150[Adim1-1],
           150);
      }
      for(Adim1 = WLa.LocGroupContext_MA + 1; Adim1 <= 9; ++(Adim1))
      {
        Cyyy9141Ia.ImpGContextDontChangeTextText150[Adim1-1] = 
"                                                                                                                                                      "
          ;
      }
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.CYYY9141).Assembly,
      	"GEN.ORT.YYY.CYYY9141",
      	"Execute",
      	Cyyy9141Ia,
      	Cyyy9141Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020147" );
      Globdata.GetStateData().SetCurrentABName( "CYYY0321_TYPE_READ" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000030" );
      WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Cyyy9141Oa.ExpContextIyy1ComponentContextString, 512);
      WLa.LocErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy9141Oa.ExpErrorIyy1ComponentSeverityCode, 1);
      WLa.LocErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Cyyy9141Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
      WLa.LocErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Cyyy9141Oa.ExpErrorIyy1ComponentOriginServid);
      WLa.LocErrorIyy1ComponentContextString = StringAttr.ValueOf(Cyyy9141Oa.ExpErrorIyy1ComponentContextString, 512);
      WLa.LocErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Cyyy9141Oa.ExpErrorIyy1ComponentReturnCode);
      WLa.LocErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Cyyy9141Oa.ExpErrorIyy1ComponentReasonCode);
      WLa.LocErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Cyyy9141Oa.ExpErrorIyy1ComponentChecksum, 15);
      Cyyy9141Ia.FreeInstance(  );
      Cyyy9141Ia = null;
      Cyyy9141Oa.FreeInstance(  );
      Cyyy9141Oa = null;
      
      Globdata.GetStateData().SetLastStatementNumber( "0000000032" );
      if ( ((double) WLa.LocErrorIyy1ComponentReturnCode < (double) WLa.LocDontChangeReturnCodesQ1Ok) )
      {
        Globdata.GetStateData().SetLastStatementNumber( "0000000033" );
        WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(WLa.LocErrorIyy1ComponentSeverityCode, 1);
        WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(WLa.LocErrorIyy1ComponentRollbackIndicator, 1);
        WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(WLa.LocErrorIyy1ComponentOriginServid);
        WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(WLa.LocErrorIyy1ComponentContextString, 512);
        WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(WLa.LocErrorIyy1ComponentReturnCode);
        WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(WLa.LocErrorIyy1ComponentReasonCode);
        WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(WLa.LocErrorIyy1ComponentChecksum, 15);
      }
      
      Globdata.GetStateData().SetLastStatementNumber( "0000000034" );
      WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf("E", 1);
      return;
    }
    
    public void f_20972166(  )
    {
      func_0020972166_esc_flag = false;
      {
        Globdata.GetStateData().SetLastStatementNumber( "0000000025" );
        WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(WLa.LocErrorIyy1ComponentSeverityCode, 1);
        WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(WLa.LocErrorIyy1ComponentRollbackIndicator, 1);
        WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(WLa.LocErrorIyy1ComponentOriginServid);
        WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(WLa.LocErrorIyy1ComponentContextString, 512);
        WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(WLa.LocErrorIyy1ComponentReturnCode);
        WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(WLa.LocErrorIyy1ComponentReasonCode);
        WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(WLa.LocErrorIyy1ComponentChecksum, 15);
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
        //    !!!!!!!!!!!                                                     
        //    !! ATTENTION : In D AB, if context string must be formed        
        //    ESCAPE AB must be exited.                                       
        //                                                                    
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        Globdata.GetStateData().SetLastStatementNumber( "0000000027" );
        func_0020972166_esc_flag = true;
        func_0020972742_esc_flag = true;
      }
      f_0020972166_esctag: 
      ;
      return;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020147_localAlloc( String abname )
    {
      // Request localview allocation 
      WLa = (GEN.ORT.YYY.CYYY0321_LA)(IefRuntimeParm2.GetInstance( GetType().Assembly,
      	"GEN.ORT.YYY.CYYY0321_LA" ));
      if ( WLa == null )
      {
        Globdata.GetStateData().SetCurrentABId( "0022020147" );
        Globdata.GetStateData().SetCurrentABName( abname );
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
        Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusIefAllocationError );
      }
    }
    
    public void f_22020147_init(  )
    {
      if ( NestingLevel < 2 )
      {
        WLa.Reset();
      }
      WLa.LocGroupContext_MA = 0;
      for(LocGroupContext_PS_001 = 1; LocGroupContext_PS_001 <= 9; ++(LocGroupContext_PS_001))
      {
      }
      WOa.ExpTypeTinstanceId = "00000000000000000000";
      WOa.ExpTypeTreferenceId = "00000000000000000000";
      WOa.ExpTypeTcreateUserId = "        ";
      WOa.ExpTypeTupdateUserId = "        ";
      WOa.ExpTypeTkeyAttrText = "    ";
      WOa.ExpTypeTsearchAttrText = "                    ";
      WOa.ExpTypeTotherAttrText = "  ";
      WOa.ExpTypeTotherAttrDate = 00000000;
      WOa.ExpTypeTotherAttrTime = 00000000;
      WOa.ExpTypeTotherAttrAmount = DecimalAttr.GetDefaultValue();
      WOa.ExpErrorIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorIyy1ComponentRollbackIndicator = " ";
      WOa.ExpErrorIyy1ComponentOriginServid = 0.0;
      WOa.ExpErrorIyy1ComponentContextString = "";
      WOa.ExpErrorIyy1ComponentReturnCode = 0;
      WOa.ExpErrorIyy1ComponentReasonCode = 0;
      WOa.ExpErrorIyy1ComponentChecksum = "               ";
      LocGroupContext_PS_001 = 1;
    }
    public void f_173015126_rgvc(  )
    {
      if ( (LocGroupContext_PS_001 > 9) || (LocGroupContext_PS_001 < 1) )
      {
        Globdata.GetErrorData().SetViewOverflow( ErrorData.ErrorEncounteredErrorFound );
        {
          throw new ABException();
        }
      }
    }
  }// end class
  
}
